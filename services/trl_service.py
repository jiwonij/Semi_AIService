from __future__ import annotations

import json
import re
from typing import Dict, List, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from utils.config import (
    TRL_CONFIDENCE_THRESHOLD,
    TRL_MIN_CITATION_COUNT,
    TRL_MIN_EVIDENCE_TYPE_COUNT,
)
from services.retrieval_service import RetrievalService


TRL_SYSTEM = """
You are evaluating the technology maturity of one company and one technology.

You must rely only on the provided evidence block.

Core principle:
- In semiconductor industry, TRL 4–6 is the most common stage and often lacks public production evidence.
- Do NOT assign low TRL just because production or shipment evidence is missing.

Interpretation rules:

- TRL 1–3:
  Only research-level evidence (papers, patents, academic work).
  No company-level development activity.

- TRL 4–5:
  Company is actively developing the technology.
  Evidence includes architecture design, roadmap, engineering discussion.

- TRL 5–6:
  Prototype, validation, benchmarking, or system-level testing.

- TRL 7–9:
  Requires clear production, shipment, customer deployment.

Critical rules:
- Absence of production evidence does NOT mean low TRL.
- If company-level development exists → minimum TRL 4.
- If engineering validation exists → TRL 5–6.
- Only assign TRL 1–3 if evidence is purely academic.

- Reasoning MUST cite citation_id.
- Ignore evidence unrelated to the target technology.

Return JSON:
{
  "trl_range": "string",
  "confidence": float,
  "reasoning": "string with citation_id references",
  "signals": ["string"],
  "sources": ["citation_id"]
}
""".strip()


class TRLService:
    TECHNOLOGIES = ["HBM4", "PIM", "CXL", "Advanced Packaging"]
    COMPANIES = ["SK hynix", "Samsung", "Micron"]

    TECH_KEYWORDS = {
        "HBM4": [
            "hbm4",
            "hbm 4",
            "12hi",
            "12-high",
            "12-high stack",
            "high bandwidth memory 4",
        ],
        "PIM": [
            "pim",
            "processing-in-memory",
            "processing in memory",
            "in-memory",
            "aim",
            "accelerator in memory",
        ],
        "CXL": [
            "cxl",
            "compute express link",
            "memory expander",
            "cxl memory",
            "cxl controller",
        ],
        "Advanced Packaging": [
            "advanced packaging",
            "2.5d",
            "3d packaging",
            "3d stack",
            "tsv",
            "interposer",
            "hybrid bonding",
            "chiplet",
            "packaging",
            "heterogeneous integration",
        ],
    }

    NEGATIVE_TECH_KEYWORDS = {
        "HBM4": [
            "pim",
            "processing-in-memory",
            "processing in memory",
            "aim",
            "cxl",
            "compute express link",
        ],
        "PIM": [
            "hbm4",
            "hbm 4",
            "cxl",
            "compute express link",
        ],
        "CXL": [
            "hbm4",
            "hbm 4",
            "pim",
            "processing-in-memory",
            "processing in memory",
            "aim",
        ],
        "Advanced Packaging": [],
    }

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.retrieval_service = RetrievalService()

    def _clean_text(self, value: Any) -> str:
        text = str(value) if value is not None else ""
        text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        text = text.replace("\x00", " ")
        return text.strip()

    def run(self, validated_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        trl_comparison: Dict[str, Dict[str, Any]] = {}

        for tech in self.TECHNOLOGIES:
            tech_block: Dict[str, Any] = {}

            for company in self.COMPANIES:
                docs = self._retrieve(company, tech)
                tech_block[company] = self._analyze(company, tech, docs)

            tech_block["comparison_summary"] = self._compare(tech, tech_block)
            trl_comparison[tech] = tech_block

        trl_result = {
            "trl_comparison": trl_comparison,
            "overall_limitation_note": (
                "TRL은 회사·기술별 검색 결과를 바탕으로 공개 자료 수준에서 추정했다. "
                "직접적인 양산·출하·고객 적용 근거가 부족한 경우에는 시제품, 검증, 샘플링, "
                "로드맵, 투자, 연구 활동 등 간접 신호를 보수적으로 반영했다."
            ),
        }

        metrics = self._evaluate(trl_result, validated_evidence)

        return {
            "trl_result": trl_result,
            "metrics": metrics,
        }

    def _retrieve(self, company: str, tech: str) -> List[Dict[str, Any]]:
        queries = [
            f"{company} {tech} development status production roadmap",
            f"{company} {tech} commercialization manufacturing roadmap",
            f"{company} {tech} product launch package memory roadmap",
        ]

        retrieval_result = self.retrieval_service.run(queries)
        chunks = retrieval_result.get("retrieved_evidence", [])

        filtered_chunks = self._filter_chunks_by_technology(chunks, tech)

        filtered_chunks = sorted(
            filtered_chunks,
            key=lambda x: float(x.get("score", 0.0) or 0.0),
            reverse=True,
        )[:5]

        docs: List[Dict[str, Any]] = []
        for i, chunk in enumerate(filtered_chunks, 1):
            docs.append(
                {
                    "citation_id": f"{company}_{tech}_{i}",
                    "title": self._clean_text(chunk.get("title", "")),
                    "url": self._clean_text(chunk.get("url", "")),
                    "content": self._clean_text(chunk.get("content", "")),
                    "source_type": self._clean_text(
                        chunk.get("source_type", "internal_document")
                    ),
                    "score": float(chunk.get("score", 0.0) or 0.0),
                    "source_name": self._clean_text(chunk.get("source_name", "")),
                    "metadata": chunk.get("metadata", {}),
                }
            )

        return docs

    def _filter_chunks_by_technology(
        self,
        chunks: List[Dict[str, Any]],
        tech: str,
    ) -> List[Dict[str, Any]]:
        positive_keywords = [
            keyword.lower() for keyword in self.TECH_KEYWORDS.get(tech, [])
        ]
        negative_keywords = [
            keyword.lower() for keyword in self.NEGATIVE_TECH_KEYWORDS.get(tech, [])
        ]

        filtered: List[Dict[str, Any]] = []

        for chunk in chunks:
            title = self._clean_text(chunk.get("title", "")).lower()
            content = self._clean_text(chunk.get("content", "")).lower()
            source_name = self._clean_text(chunk.get("source_name", "")).lower()
            joined = f"{title} {source_name} {content}"

            positive_hits = sum(1 for keyword in positive_keywords if keyword in joined)
            negative_hits = sum(1 for keyword in negative_keywords if keyword in joined)

            if tech == "Advanced Packaging":
                packaging_related = any(keyword in joined for keyword in positive_keywords)
                if packaging_related:
                    filtered.append(chunk)
                continue

            if positive_hits == 0:
                continue

            if negative_hits > positive_hits:
                continue

            filtered.append(chunk)

        if filtered:
            return filtered

        # 기술 키워드 필터 후 아무 것도 안 남으면, 상위 결과를 그대로 쓰지 않고
        # 가장 약한 형태로라도 기술명이 포함된 결과만 재시도
        fallback: List[Dict[str, Any]] = []
        tech_name = tech.lower()
        for chunk in chunks:
            title = self._clean_text(chunk.get("title", "")).lower()
            content = self._clean_text(chunk.get("content", "")).lower()
            joined = f"{title} {content}"
            if tech_name in joined:
                fallback.append(chunk)

        return fallback

    def _analyze(self, company: str, tech: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(docs) < 2:
            joined = " ".join(self._clean_text(d.get("content", "")).lower() for d in docs)

            # 회사 개발 흔적 있으면 최소 TRL 4
            if any(k in joined for k in [
                "roadmap", "architecture", "design", "development", "engineering"
            ]):
                return {
                    "trl_level": 4,
                    "trl_confidence": 0.4,
                    "assessment": (
                        f"{company}의 {tech}는 공개 근거는 제한적이나 "
                        f"개발 및 설계 활동이 확인되어 최소 중간 단계로 판단된다."
                    ),
                    "indirect_signals_used": ["개발 활동 존재"],
                    "citations": [],
                }

            # 진짜 아무 것도 없을 때만 TRL 1
            return {
                "trl_level": 1,
                "trl_confidence": 0.2,
                "assessment": (
                    f"{company}의 {tech} 관련 근거가 거의 없어 초기 개념 수준으로 평가했다."
                ),
                "indirect_signals_used": ["근거 부족"],
                "citations": [],
            }

        evidence = self._format_evidence(company, tech, docs)
        llm_out = self._call_llm(company, tech, evidence)

        trl_level = self._parse_range(llm_out.get("trl_range", "1-2"))
        confidence = self._safe_float(llm_out.get("confidence", 0.4), default=0.4)

        reasoning = self._clean_text(llm_out.get("reasoning", ""))
        signals = llm_out.get("signals", [])
        sources = llm_out.get("sources", [])

        if not isinstance(signals, list):
            signals = []
        if not isinstance(sources, list):
            sources = []

        source_ids = self._filter_valid_source_ids(sources, docs)
        citations = self._map_sources(source_ids, docs)

        if not self._reasoning_has_citation(reasoning, source_ids):
            reasoning = self._append_reasoning_citations(reasoning, source_ids)

        trl_level = self._apply_trl_sanity_check(
            trl_level=trl_level,
            citations=source_ids,
            docs=docs,
        )

        confidence = self._adjust_confidence(
            confidence=confidence,
            trl_level=trl_level,
            citation_count=len(source_ids),
            doc_count=len(docs),
        )

        if not reasoning:
            reasoning = self._fallback_assessment(company, tech, trl_level, source_ids)

        if not signals:
            signals = self._derive_basic_signals(docs)

        return {
            "trl_level": trl_level,
            "trl_confidence": round(confidence, 4),
            "assessment": reasoning,
            "indirect_signals_used": signals[:5],
            "citations": citations[:5],
        }

    def _format_evidence(self, company: str, tech: str, docs: List[Dict[str, Any]]) -> str:
        lines = [
            f"Company: {self._clean_text(company)}",
            f"Technology: {self._clean_text(tech)}",
            "Evidence:",
        ]

        for d in docs:
            title = self._clean_text(d.get("title", ""))[:200]
            source_type = self._clean_text(d.get("source_type", ""))[:50]
            content = self._clean_text(d.get("content", ""))[:600]
            score = d.get("score", 0.0)

            lines.append(
                f"- citation_id: {d['citation_id']}\n"
                f"  title: {title}\n"
                f"  source_type: {source_type}\n"
                f"  score: {score:.4f}\n"
                f"  text: {content}"
            )

        evidence = "\n".join(lines)
        return self._clean_text(evidence)[:6000]

    def _call_llm(self, company: str, tech: str, evidence: str) -> Dict[str, Any]:
        prompt = (
            f"Evaluate the maturity of the following pair.\n\n"
            f"Company: {company}\n"
            f"Technology: {tech}\n\n"
            f"{evidence}"
        )

        res = self.llm.invoke(
            [
                SystemMessage(content=TRL_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )

        raw = self._clean_text(res.content)
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            return json.loads(raw)
        except Exception:
            return {
                "trl_range": "1-2",
                "confidence": 0.25,
                "reasoning": "",
                "signals": [],
                "sources": [],
            }

    def _parse_range(self, trl_range: str) -> int:
        text = self._clean_text(trl_range)
        nums = re.findall(r"\d+", text)
        if not nums:
            return 1
        return max(1, min(9, int(nums[-1])))

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _filter_valid_source_ids(
        self,
        source_ids: List[str],
        docs: List[Dict[str, Any]],
    ) -> List[str]:
        valid_ids = {d["citation_id"] for d in docs}
        result: List[str] = []

        for source_id in source_ids:
            cleaned = self._clean_text(source_id)
            if cleaned in valid_ids and cleaned not in result:
                result.append(cleaned)

        return result

    def _map_sources(self, source_ids: List[str], docs: List[Dict[str, Any]]) -> List[str]:
        mapping = {
            d["citation_id"]: d["url"] or d["source_name"] or d["title"]
            for d in docs
        }
        return [mapping[s] for s in source_ids if s in mapping]

    def _reasoning_has_citation(self, reasoning: str, source_ids: List[str]) -> bool:
        return any(source_id in reasoning for source_id in source_ids)

    def _append_reasoning_citations(self, reasoning: str, source_ids: List[str]) -> str:
        citation_tail = ", ".join(source_ids[:3])
        base = self._clean_text(reasoning)

        if not citation_tail:
            return base

        if not base:
            return f"평가 근거는 {citation_tail}에 기반한다."

        return f"{base} Evidence used: {citation_tail}."

    def _apply_trl_sanity_check(
        self,
        trl_level: int,
        citations: List[str],
        docs: List[Dict[str, Any]],
    ) -> int:
        joined = " ".join(self._clean_text(d.get("content", "")).lower() for d in docs)

        has_production_signal = any(
            keyword in joined
            for keyword in [
                "mass production",
                "high-volume production",
                "volume production",
                "shipment",
                "customer deployment",
                "commercial production",
                "commercial launch",
            ]
        )

        if trl_level >= 8 and (len(citations) < 2 or not has_production_signal):
            return 7

        if trl_level >= 7 and not has_production_signal:
            return 6
        if trl_level < 4:
            if any(k in joined for k in [
                "roadmap", "architecture", "design", "development", "engineering"
            ]):
                return 4

        return trl_level

    def _adjust_confidence(
        self,
        confidence: float,
        trl_level: int,
        citation_count: int,
        doc_count: int,
    ) -> float:
        adjusted = max(0.0, min(0.95, confidence))

        if citation_count < 2:
            adjusted -= 0.15
        elif citation_count >= 3:
            adjusted += 0.05

        if doc_count < 3:
            adjusted -= 0.05

        if trl_level >= 7 and citation_count < 2:
            adjusted -= 0.1

        return max(0.1, min(0.95, adjusted))

    def _fallback_assessment(
        self,
        company: str,
        tech: str,
        trl_level: int,
        source_ids: List[str],
    ) -> str:
        phase = self._trl_phase_label(trl_level)
        if source_ids:
            return (
                f"{company}의 {tech}는 {phase} 수준으로 해석된다. "
                f"직접 근거는 제한적이며 판단에는 {', '.join(source_ids[:3])}가 활용되었다."
            )
        return (
            f"{company}의 {tech}는 확보된 공개 근거가 제한적이어서 "
            f"{phase} 수준으로 보수적으로 해석된다."
        )

    def _trl_phase_label(self, trl_level: int) -> str:
        if 1 <= trl_level <= 3:
            return "초기 연구·개념 검증 단계"
        if 4 <= trl_level <= 6:
            return "시제품·검증 중심의 중간 성숙 단계"
        return "상용화 준비 또는 시장 적용에 가까운 단계"

    def _derive_basic_signals(self, docs: List[Dict[str, Any]]) -> List[str]:
        joined = " ".join(self._clean_text(d.get("content", "")).lower() for d in docs)
        signals: List[str] = []

        keyword_map = {
            "prototype": "시제품 또는 프로토타입 신호",
            "validation": "검증 또는 평가 신호",
            "sample": "샘플링 신호",
            "qualification": "품질 인증 또는 qualification 신호",
            "roadmap": "로드맵 기반 간접 신호",
            "patent": "특허 기반 간접 신호",
            "paper": "논문 또는 연구 발표 신호",
            "shipment": "출하 또는 고객 전달 신호",
            "production": "생산 관련 신호",
        }

        for keyword, label in keyword_map.items():
            if keyword in joined and label not in signals:
                signals.append(label)

        return signals or ["공개 자료 기반 간접 신호"]

    def _compare(self, technology: str, tech_block: Dict[str, Any]) -> str:
        ranking = sorted(
            [
                (
                    company,
                    tech_block[company]["trl_level"],
                    tech_block[company]["trl_confidence"],
                )
                for company in self.COMPANIES
            ],
            key=lambda x: (x[1], x[2]),
            reverse=True,
        )

        leader = ranking[0][0]
        trailer = ranking[-1][0]
        leader_level = ranking[0][1]
        trailer_level = ranking[-1][1]

        if leader_level == trailer_level:
            return (
                f"{technology} 기준으로 세 회사의 성숙도 차이가 크지 않으며, "
                f"공개 근거상 뚜렷한 우위를 단정하기 어렵다."
            )

        return (
            f"{technology} 기준으로는 {leader}가 상대적으로 높은 성숙도를 보이며, "
            f"{trailer}는 비교적 보수적인 단계로 평가된다."
        )

    def _evaluate(
        self,
        trl_result: Dict[str, Any],
        validated_evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        confidences: List[float] = []
        citations = 0

        for tech in trl_result["trl_comparison"].values():
            for company in self.COMPANIES:
                confidences.append(float(tech[company]["trl_confidence"]))
                citations += len(tech[company]["citations"])

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        evidence_type_count = len(
            {
                d.get("source_type")
                for d in validated_evidence
                if d.get("source_type")
            }
        )

        return {
            "trl_confidence": round(avg_conf, 4),
            "evidence_type_count": evidence_type_count,
            "citation_count": citations,
            "pass": (
                avg_conf >= TRL_CONFIDENCE_THRESHOLD
                and citations >= TRL_MIN_CITATION_COUNT
                and evidence_type_count >= TRL_MIN_EVIDENCE_TYPE_COUNT
            ),
        }
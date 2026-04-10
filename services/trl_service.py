import re
from typing import Dict, List, Any

from utils.config import (
    TRL_CONFIDENCE_THRESHOLD,
    TRL_MIN_CITATION_COUNT,
    TRL_MIN_EVIDENCE_TYPE_COUNT,
)


class TRLService:
    """
    검증된 근거를 바탕으로 반도체 기술의 TRL을 규칙 기반으로 추정하는 서비스.
    공개 정보만으로 직접 검증이 어려운 구간은 confidence와 limitation note로 보완한다.
    """

    TECHNOLOGIES = ["HBM4", "PIM", "CXL", "Advanced Packaging"]
    COMPANIES = ["SK hynix", "Samsung", "Micron"]

    TECH_KEYWORDS = {
        "HBM4": ["hbm4", "12hi", "12-high", "high bandwidth memory"],
        "PIM": ["pim", "processing-in-memory", "processing in memory", "in-memory"],
        "CXL": ["cxl", "compute express link", "memory expander"],
        "Advanced Packaging": [
            "advanced packaging",
            "2.5d",
            "3d packaging",
            "tsv",
            "interposer",
            "hybrid bonding",
            "chiplet",
        ],
    }

    COMPANY_KEYWORDS = {
        "SK hynix": ["sk hynix", "skhynix", "hynix"],
        "Samsung": ["samsung", "samsung electronics"],
        "Micron": ["micron", "micron technology"],
    }

    SOURCE_TYPE_BASE_TRL = {
        "paper": 2,
        "patent": 3,
        "news": 5,
        "article": 5,
        "blog": 4,
        "press_release": 6,
        "report": 5,
        "web": 5,
        "official": 6,
    }

    PRODUCTION_KEYWORDS = [
        "mass production",
        "volume production",
        "customer adoption",
        "shipment",
        "sampling",
        "sample",
        "qualification",
        "production ready",
        "production",
    ]

    PROTOTYPE_KEYWORDS = [
        "prototype",
        "demonstration",
        "demo",
        "benchmark",
        "evaluation",
        "test chip",
        "fpga",
        "platform",
        "validation",
    ]

    INDIRECT_SIGNAL_KEYWORDS = [
        "roadmap",
        "hiring",
        "recruiting",
        "investment",
        "capex",
        "conference",
        "publication",
        "patent filing",
        "partnership",
    ]

    TRL_CAP = {
        "HBM4": 7,
        "PIM": 6,
        "CXL": 7,
        "Advanced Packaging": 8,
    }

    def run(self, validated_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        trl_comparison: Dict[str, Dict[str, Any]] = {}

        for technology in self.TECHNOLOGIES:
            tech_block: Dict[str, Any] = {}

            for company in self.COMPANIES:
                matched_docs = self._filter_evidence(validated_evidence, company, technology)
                tech_block[company] = self._build_company_trl(company, technology, matched_docs)

            tech_block["comparison_summary"] = self._build_comparison_summary(technology, tech_block)
            trl_comparison[technology] = tech_block

        trl_result = {
            "trl_comparison": trl_comparison,
            "overall_limitation_note": (
                "TRL 4~6 구간은 공개 정보만으로 직접 검증하기 어려운 추정 영역이다. "
                "따라서 특허, 논문, 로드맵, 채용, 투자, 시제품 검증 및 발표 자료 등 간접 신호를 종합하여 판단했다."
            ),
        }

        metrics = self._evaluate_trl(trl_result, validated_evidence)

        return {
            "trl_result": trl_result,
            "metrics": metrics,
        }

    def _filter_evidence(
        self,
        validated_evidence: List[Dict[str, Any]],
        company: str,
        technology: str,
    ) -> List[Dict[str, Any]]:
        company_keywords = self.COMPANY_KEYWORDS[company]
        tech_keywords = self.TECH_KEYWORDS[technology]

        matched = []
        for doc in validated_evidence:
            text = self._doc_text(doc)

            has_company = any(keyword in text for keyword in company_keywords)
            has_tech = any(keyword in text for keyword in tech_keywords)

            if has_company and has_tech:
                matched.append(doc)

        return matched

    def _build_company_trl(
        self,
        company: str,
        technology: str,
        matched_docs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not matched_docs:
            return {
                "trl_level": 3,
                "trl_confidence": 0.45,
                "assessment": (
                    f"{company}의 {technology} 관련 공개 근거가 제한적이어서 "
                    "초기 연구 또는 개념 검증 단계 중심으로 보수적으로 추정했다."
                ),
                "indirect_signals_used": ["공개 근거 부족"],
                "citations": [],
            }

        source_types = set()
        citations = []
        corpus_parts = []

        for doc in matched_docs:
            source_type = str(doc.get("source_type", "")).strip().lower()
            if source_type:
                source_types.add(source_type)

            url = str(doc.get("url", "")).strip()
            title = str(doc.get("title", "")).strip()

            if url:
                citations.append(url)
            elif title:
                citations.append(title)

            corpus_parts.append(self._doc_text(doc))

        corpus = " ".join(corpus_parts)

        base_trl = self._estimate_base_trl(source_types)
        signal_bonus, signals = self._estimate_signal_bonus(corpus)
        trl_level = min(base_trl + signal_bonus, self.TRL_CAP.get(technology, 7))
        trl_level = max(1, trl_level)

        confidence = self._estimate_confidence(
            doc_count=len(matched_docs),
            source_type_count=len(source_types),
            citation_count=len(citations),
            trl_level=trl_level,
        )

        evidence_text = self._extract_key_evidence(matched_docs)

        assessment = self._build_assessment(
            company=company,
            technology=technology,
            trl_level=trl_level,
            source_types=sorted(source_types),
            signals=signals,
            evidence_text=evidence_text,
        )

        unique_citations = self._unique_keep_order(citations)[:5]

        return {
            "trl_level": trl_level,
            "trl_confidence": confidence,
            "assessment": assessment,
            "indirect_signals_used": signals[:5] if signals else ["공개 자료 기반 추정"],
            "citations": unique_citations,
        }

    def _estimate_base_trl(self, source_types: set[str]) -> int:
        if not source_types:
            return 3

        scores = []
        for source_type in source_types:
            scores.append(self.SOURCE_TYPE_BASE_TRL.get(source_type, 5))

        return max(scores) if scores else 3

    def _estimate_signal_bonus(self, corpus: str) -> tuple[int, List[str]]:
        text = corpus.lower()
        bonus = 0
        signals: List[str] = []

        if any(keyword in text for keyword in self.PRODUCTION_KEYWORDS):
            bonus += 2
            signals.append("양산/고객 공급/샘플 관련 신호")

        if any(keyword in text for keyword in self.PROTOTYPE_KEYWORDS):
            bonus += 1
            signals.append("시제품/검증/벤치마크 신호")

        if any(keyword in text for keyword in self.INDIRECT_SIGNAL_KEYWORDS):
            bonus += 1
            signals.append("로드맵/채용/투자/학회 활동 신호")

        return bonus, signals

    def _estimate_confidence(
        self,
        doc_count: int,
        source_type_count: int,
        citation_count: int,
        trl_level: int,
    ) -> float:
        score = 0.45
        score += min(doc_count, 4) * 0.07
        score += min(source_type_count, 3) * 0.06
        score += min(citation_count, 4) * 0.04

        if 4 <= trl_level <= 6:
            score -= 0.05

        return round(min(score, 0.9), 4)

    def _extract_key_evidence(self, matched_docs: List[Dict[str, Any]]) -> str:
        candidate_sentences: List[str] = []
        keywords = (
            self.PRODUCTION_KEYWORDS
            + self.PROTOTYPE_KEYWORDS
            + self.INDIRECT_SIGNAL_KEYWORDS
        )

        for doc in matched_docs:
            content = str(doc.get("content", ""))
            if not content:
                continue

            sentences = re.split(r"[.\n]", content)
            for sentence in sentences:
                s = sentence.strip()
                if len(s) < 30:
                    continue

                s_lower = s.lower()
                if any(keyword in s_lower for keyword in keywords):
                    candidate_sentences.append(s)

        unique_sentences = self._unique_keep_order(candidate_sentences)
        if not unique_sentences:
            return "직접적인 양산·검증 표현은 제한적이나 공개 자료에서 관련 기술 개발 및 사업화 정황이 확인된다"

        return " / ".join(unique_sentences[:2])

    def _build_assessment(
        self,
        company: str,
        technology: str,
        trl_level: int,
        source_types: List[str],
        signals: List[str],
        evidence_text: str,
    ) -> str:
        source_desc = ", ".join(source_types) if source_types else "공개 자료"
        signal_desc = ", ".join(signals) if signals else "직접 양산 근거는 제한적"

        if 1 <= trl_level <= 3:
            phase = "초기 연구 및 개념 검증 단계"
        elif 4 <= trl_level <= 6:
            phase = "시제품 검증 및 추정 기반의 중간 성숙 단계"
        else:
            phase = "상용화 준비 또는 시장 적용에 가까운 단계"

        return (
            f"{company}의 {technology}는 {phase}로 판단된다. "
            f"판단 근거는 {source_desc} 기반 자료이며, {signal_desc}가 확인된다. "
            f"대표 근거로는 '{evidence_text}'가 있다. "
            f"이를 종합해 TRL을 추정했다. "
            f"특히 TRL 4~6 구간은 공개 정보만으로 직접 검증이 어려워 간접 신호를 포함해 해석했다."
        )

    def _build_comparison_summary(self, technology: str, tech_block: Dict[str, Any]) -> str:
        rows = []
        for company in self.COMPANIES:
            company_data = tech_block.get(company, {})
            level = company_data.get("trl_level", "-")
            conf = company_data.get("trl_confidence", 0.0)
            rows.append((company, level, conf))

        rows.sort(key=lambda x: (x[1], x[2]), reverse=True)

        leader = rows[0][0] if rows else "확인 불가"
        trailing = rows[-1][0] if rows else "확인 불가"

        return (
            f"{technology} 기준으로는 {leader}가 상대적으로 높은 성숙도를 보이며, "
            f"{trailing}는 공개 근거상 보다 보수적인 단계로 평가된다. "
            f"다만 중간 구간의 TRL은 공개 자료만으로 직접 검증하기 어려워 추정 오차 가능성이 있다."
        )

    def _evaluate_trl(
        self,
        trl_result: Dict[str, Any],
        validated_evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trl_comparison = trl_result.get("trl_comparison", {})

        confidence_values: List[float] = []
        citation_count = 0

        for technology_data in trl_comparison.values():
            if not isinstance(technology_data, dict):
                continue

            for company in self.COMPANIES:
                company_data = technology_data.get(company, {})
                if not isinstance(company_data, dict):
                    continue

                confidence_values.append(float(company_data.get("trl_confidence", 0.0)))
                citation_count += len(company_data.get("citations", []))

        evidence_type_count = self._count_evidence_types(validated_evidence)
        avg_confidence = round(
            sum(confidence_values) / len(confidence_values), 4
        ) if confidence_values else 0.0

        return {
            "trl_confidence": avg_confidence,
            "evidence_type_count": evidence_type_count,
            "citation_count": citation_count,
            "pass": (
                avg_confidence >= TRL_CONFIDENCE_THRESHOLD
                and evidence_type_count >= TRL_MIN_EVIDENCE_TYPE_COUNT
                and citation_count >= TRL_MIN_CITATION_COUNT
            ),
        }

    def _count_evidence_types(self, validated_evidence: List[Dict[str, Any]]) -> int:
        types = set()
        for doc in validated_evidence:
            source_type = str(doc.get("source_type", "")).strip().lower()
            if source_type:
                types.add(source_type)
        return len(types)

    def _doc_text(self, doc: Dict[str, Any]) -> str:
        title = str(doc.get("title", ""))
        content = str(doc.get("content", ""))
        url = str(doc.get("url", ""))
        return f"{title} {content} {url}".lower()

    def _unique_keep_order(self, items: List[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
        return result
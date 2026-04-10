import json
from typing import Dict, List

from openai import OpenAI

from utils.config import (
    OPENAI_API_KEY,
    TRL_CONFIDENCE_THRESHOLD,
    TRL_MIN_CITATION_COUNT,
    TRL_MIN_EVIDENCE_TYPE_COUNT,
)


# 기술별 현실적인 TRL 상한
TRL_CAP = {
    "HBM4": 7,
    "PIM": 6,
    "CXL": 6,
    "Advanced Packaging": 8,
}


class TRLService:
    """
    근거 기반으로 기술 성숙도 TRL을 추정하는 서비스.
    """

    def __init__(self) -> None:
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    def run(self, validated_evidence: List[Dict]) -> Dict:
        if not self.client:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

        trl_result = self._estimate_with_llm(validated_evidence)

        # 🔥 LLM 결과 후처리 (핵심)
        trl_result = self._postprocess_trl(trl_result)

        metrics = self._evaluate_trl(trl_result, validated_evidence)

        return {
            "trl_result": trl_result,
            "metrics": metrics,
        }

    def _estimate_with_llm(self, validated_evidence: List[Dict]) -> Dict:
        evidence_text = self._format_evidence(validated_evidence)

        system_prompt = """
You are a TRL Analysis Agent for semiconductor technology assessment.

Return JSON only.

Requirements:
- Each company MUST include at least 2 citations.
- Citations must be URLs or identifiable sources.
- Do NOT return empty citations.
- If evidence is weak, still include best available sources.

Rules:
- paper → TRL 1~3
- patent → TRL 2~4
- indirect signals → TRL 4~6
- product / mass production → TRL 7~9
"""

        user_prompt = f"""
Validated evidence:
{evidence_text}
"""

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return json.loads(response.choices[0].message.content.strip())

    # 🔥 TRL 값 안정화 + 타입 방어
    def _postprocess_trl(self, trl_result: Dict) -> Dict:
        trl_comparison = trl_result.get("trl_comparison", {})

        for tech, tech_data in trl_comparison.items():
            if not isinstance(tech_data, dict):
                continue

            cap = TRL_CAP.get(tech, 7)

            for company in ["SK hynix", "Samsung", "Micron"]:
                company_data = tech_data.get(company)

                if not isinstance(company_data, dict):
                    continue

                level = company_data.get("trl_level", 0)

                # 🔥 TRL 제한 (핵심)
                company_data["trl_level"] = max(1, min(level, cap))

        return trl_result

    # 🔥 citation 실제 개수 계산 (에러 방지 포함)
    def _count_citations(self, trl_comparison: Dict) -> int:
        count = 0

        for tech_data in trl_comparison.values():
            if not isinstance(tech_data, dict):
                continue

            for company in ["SK hynix", "Samsung", "Micron"]:
                company_data = tech_data.get(company)

                if not isinstance(company_data, dict):
                    continue

                citations = company_data.get("citations", [])
                count += len(citations)

        return count

    # 🔥 evidence 타입 개수 계산
    def _count_evidence_types(self, validated_evidence: List[Dict]) -> int:
        types = set()

        for doc in validated_evidence:
            if doc.get("source_type"):
                types.add(doc["source_type"])

        return len(types)

    def _evaluate_trl(self, trl_result: Dict, validated_evidence: List[Dict]) -> Dict:
        trl_comparison = trl_result.get("trl_comparison", {})

        evidence_type_count = self._count_evidence_types(validated_evidence)
        citation_count = self._count_citations(trl_comparison)

        confidence_values = []

        for technology_data in trl_comparison.values():
            if not isinstance(technology_data, dict):
                continue

            for company in ["SK hynix", "Samsung", "Micron"]:
                company_data = technology_data.get(company)

                if not isinstance(company_data, dict):
                    continue

                if "trl_confidence" in company_data:
                    confidence_values.append(float(company_data["trl_confidence"]))

        avg_confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values else 0.0
        )

        return {
            "trl_confidence": round(avg_confidence, 4),
            "evidence_type_count": evidence_type_count,
            "citation_count": citation_count,
            "pass": (
                avg_confidence >= TRL_CONFIDENCE_THRESHOLD
                and evidence_type_count >= TRL_MIN_EVIDENCE_TYPE_COUNT
                and citation_count >= TRL_MIN_CITATION_COUNT
            ),
        }

    def _format_evidence(self, validated_evidence: List[Dict]) -> str:
        lines = []

        for idx, doc in enumerate(validated_evidence[:25], start=1):
            lines.append(
                f"[{idx}] title={doc.get('title', '')} | "
                f"source_type={doc.get('source_type', '')} | "
                f"url={doc.get('url', '')} | "
                f"content={doc.get('content', '')[:900]}"
            )

        return "\n".join(lines)
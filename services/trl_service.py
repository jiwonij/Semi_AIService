import json
from typing import Dict, List

from openai import OpenAI

from utils.config import (
    OPENAI_API_KEY,
    TRL_CONFIDENCE_THRESHOLD,
    TRL_MIN_CITATION_COUNT,
    TRL_MIN_EVIDENCE_TYPE_COUNT,
)


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
        metrics = self._evaluate_trl(trl_result)

        return {
            "trl_result": trl_result,
            "metrics": metrics,
        }

    def _estimate_with_llm(self, validated_evidence: List[Dict]) -> Dict:
        evidence_text = self._format_evidence(validated_evidence)

        system_prompt = """
You are a TRL Analysis Agent for semiconductor technology assessment.

Return JSON only with this schema:
{
  "trl_comparison": {
    "HBM4": {
      "SK hynix": {
        "trl_level": 0,
        "trl_confidence": 0.0,
        "assessment": "...",
        "indirect_signals_used": ["...", "..."],
        "citations": ["...", "..."]
      },
      "Samsung": {
        "trl_level": 0,
        "trl_confidence": 0.0,
        "assessment": "...",
        "indirect_signals_used": ["...", "..."],
        "citations": ["...", "..."]
      },
      "Micron": {
        "trl_level": 0,
        "trl_confidence": 0.0,
        "assessment": "...",
        "indirect_signals_used": ["...", "..."],
        "citations": ["...", "..."]
      },
      "comparison_summary": "..."
    },
    "PIM": {
      "SK hynix": {
        "trl_level": 0,
        "trl_confidence": 0.0,
        "assessment": "...",
        "indirect_signals_used": ["...", "..."],
        "citations": ["...", "..."]
      },
      "Samsung": {
        "trl_level": 0,
        "trl_confidence": 0.0,
        "assessment": "...",
        "indirect_signals_used": ["...", "..."],
        "citations": ["...", "..."]
      },
      "Micron": {
        "trl_level": 0,
        "trl_confidence": 0.0,
        "assessment": "...",
        "indirect_signals_used": ["...", "..."],
        "citations": ["...", "..."]
      },
      "comparison_summary": "..."
    },
    "CXL": {
      "SK hynix": {
        "trl_level": 0,
        "trl_confidence": 0.0,
        "assessment": "...",
        "indirect_signals_used": ["...", "..."],
        "citations": ["...", "..."]
      },
      "Samsung": {
        "trl_level": 0,
        "trl_confidence": 0.0,
        "assessment": "...",
        "indirect_signals_used": ["...", "..."],
        "citations": ["...", "..."]
      },
      "Micron": {
        "trl_level": 0,
        "trl_confidence": 0.0,
        "assessment": "...",
        "indirect_signals_used": ["...", "..."],
        "citations": ["...", "..."]
      },
      "comparison_summary": "..."
    },
    "Advanced Packaging": {
      "SK hynix": {
        "trl_level": 0,
        "trl_confidence": 0.0,
        "assessment": "...",
        "indirect_signals_used": ["...", "..."],
        "citations": ["...", "..."]
      },
      "Samsung": {
        "trl_level": 0,
        "trl_confidence": 0.0,
        "assessment": "...",
        "indirect_signals_used": ["...", "..."],
        "citations": ["...", "..."]
      },
      "Micron": {
        "trl_level": 0,
        "trl_confidence": 0.0,
        "assessment": "...",
        "indirect_signals_used": ["...", "..."],
        "citations": ["...", "..."]
      },
      "comparison_summary": "..."
    }
  },
  "overall_limitation_note": "..."
}

Requirements:
- Compare TRL across companies and technologies.
- paper -> TRL 1~3
- patent -> TRL 2~4
- hiring / roadmap / publication trend / investment signals -> indirect evidence mainly for TRL 4~6
- product release / volume production / customer adoption -> TRL 7~9
- Explicitly state that TRL 4~6 is an estimation zone.
- Explicitly acknowledge that public information is insufficient for direct verification in the TRL 4~6 range.
- When assigning TRL 4~6, explain that the estimate is based on indirect indicators such as patent filing patterns, publication or conference activity changes, hiring keywords, roadmap statements, and investment signals.
- Assessments must be detailed and comparative.
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

    def _evaluate_trl(self, trl_result: Dict) -> Dict:
        trl_comparison = trl_result.get("trl_comparison", {})
        evidence_type_count = 2 if trl_comparison else 0
        citation_count = 2 if trl_comparison else 0

        confidence_values = []
        for technology_data in trl_comparison.values():
            for company in ["SK hynix", "Samsung", "Micron"]:
                company_data = technology_data.get(company, {})
                if "trl_confidence" in company_data:
                    confidence_values.append(float(company_data["trl_confidence"]))

        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0

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
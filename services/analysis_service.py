import json
from typing import Dict, List

from openai import OpenAI

from utils.config import (
    ANALYSIS_COVERAGE_THRESHOLD,
    ANALYSIS_MIN_COMPANIES,
    ANALYSIS_MIN_DIFFERENTIATORS,
    ANALYSIS_MIN_IMPLICATIONS,
    ANALYSIS_MIN_THREATS,
    OPENAI_API_KEY,
)


class AnalysisService:
    """
    검증된 근거를 바탕으로 경쟁사 기술 비교와 전략적 시사점을 도출하는 서비스.
    """

    def __init__(self) -> None:
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    def run(
        self,
        validated_evidence: List[Dict],
        target_companies: List[str],
        target_technologies: List[str],
    ) -> Dict:
        if not self.client:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

        analysis_result = self._analyze_with_llm(
            validated_evidence=validated_evidence,
            target_companies=target_companies,
            target_technologies=target_technologies,
        )
        metrics = self._evaluate_analysis(analysis_result, target_companies)

        return {
            "analysis_result": analysis_result,
            "metrics": metrics,
        }

    def _analyze_with_llm(
        self,
        validated_evidence: List[Dict],
        target_companies: List[str],
        target_technologies: List[str],
    ) -> Dict:
        evidence_text = self._format_evidence(validated_evidence)

        system_prompt = """
You are an Analysis Agent for semiconductor technology strategy analysis.

The analysis must be centered on SK hynix and its competitors.
The goal is not a short summary. The goal is a deep comparative technical analysis.

Return JSON only with this schema:
{
  "technology_explanations": {
    "HBM4": "...",
    "PIM": "...",
    "CXL": "...",
    "Advanced Packaging": "..."
  },
  "competitor_comparison": {
    "SK hynix": {
      "overall_position": "...",
      "technology_strengths": ["...", "..."],
      "technology_weaknesses": ["...", "..."],
      "key_risks": ["...", "..."]
    },
    "Samsung": {
      "overall_position": "...",
      "technology_strengths": ["...", "..."],
      "technology_weaknesses": ["...", "..."],
      "key_risks": ["...", "..."]
    },
    "Micron": {
      "overall_position": "...",
      "technology_strengths": ["...", "..."],
      "technology_weaknesses": ["...", "..."],
      "key_risks": ["...", "..."]
    }
  },
  "technology_comparison": {
    "HBM4": {
      "SK hynix": "...",
      "Samsung": "...",
      "Micron": "...",
      "comparison_summary": "...",
      "threat_level_to_SK_hynix": "low/medium/high"
    },
    "PIM": {
      "SK hynix": "...",
      "Samsung": "...",
      "Micron": "...",
      "comparison_summary": "...",
      "threat_level_to_SK_hynix": "low/medium/high"
    },
    "CXL": {
      "SK hynix": "...",
      "Samsung": "...",
      "Micron": "...",
      "comparison_summary": "...",
      "threat_level_to_SK_hynix": "low/medium/high"
    },
    "Advanced Packaging": {
      "SK hynix": "...",
      "Samsung": "...",
      "Micron": "...",
      "comparison_summary": "...",
      "threat_level_to_SK_hynix": "low/medium/high"
    }
  },
  "threat_factors": ["...", "...", "..."],
  "differentiators": ["...", "...", "..."],
  "strategic_implications": ["...", "...", "..."]
}

Requirements:
- Explain each technology in enough detail for an R&D reader.
- Compare competitors technology-by-technology, not just company-by-company.
- Clearly state where SK hynix leads, where it is vulnerable, and where competitors are catching up.
- Write substantial content, not one-line summaries.
- Use evidence-backed reasoning only.
"""

        user_prompt = f"""
Target companies:
{json.dumps(target_companies, ensure_ascii=False)}

Target technologies:
{json.dumps(target_technologies, ensure_ascii=False)}

Validated evidence:
{evidence_text}
"""

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return json.loads(response.choices[0].message.content.strip())

    def _evaluate_analysis(self, analysis_result: Dict, target_companies: List[str]) -> Dict:
        comparison = analysis_result.get("competitor_comparison", {})
        threat_count = len(analysis_result.get("threat_factors", []))
        differentiator_count = len(analysis_result.get("differentiators", []))
        implication_count = len(analysis_result.get("strategic_implications", []))

        matched_companies = 0
        for company in target_companies:
            if company in comparison and comparison.get(company):
                matched_companies += 1

        coverage_score = round(matched_companies / max(1, len(target_companies)), 4)

        return {
            "coverage_score": coverage_score,
            "company_count": len(target_companies),
            "threat_count": threat_count,
            "differentiator_count": differentiator_count,
            "implication_count": implication_count,
            "pass": (
                coverage_score >= ANALYSIS_COVERAGE_THRESHOLD
                and len(target_companies) >= ANALYSIS_MIN_COMPANIES
                and threat_count >= ANALYSIS_MIN_THREATS
                and differentiator_count >= ANALYSIS_MIN_DIFFERENTIATORS
                and implication_count >= ANALYSIS_MIN_IMPLICATIONS
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
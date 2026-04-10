from typing import Dict, List

from openai import OpenAI

from utils.config import (
    OPENAI_API_KEY,
    VALIDATION_CREDIBILITY_THRESHOLD,
    VALIDATION_CONSISTENCY_THRESHOLD,
    WEB_FACT_OVERLAP_THRESHOLD,
    WEB_CONTRADICTION_THRESHOLD,
)


class ValidationService:
    """
    Retrieval + WebSearch 결과를 기반으로 근거 검증을 수행하는 서비스.
    """

    def __init__(self) -> None:
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    def run(
        self,
        retrieved_evidence: List[Dict],
        raw_documents: List[Dict],
    ) -> Dict:

        if not self.client:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

        # 1. Retrieval + Web 결과 통합
        merged_documents = retrieved_evidence + raw_documents

        if not merged_documents:
            return {
                "validated_evidence": [],
                "conflicting_evidence": [],
                "metrics": self._empty_metrics(),
            }

        # 2. score 기준 정렬 (없으면 0)
        ranked_documents = sorted(
            merged_documents,
            key=lambda x: x.get("score", 0),
            reverse=True,
        )[:15]  # 너무 많이 넣으면 LLM 과부하

        # 3. LLM 검증
        validation_result = self._validate_with_llm(ranked_documents)

        validated_evidence = validation_result.get("validated_evidence", [])
        conflicting_evidence = validation_result.get("conflicting_evidence", [])

        # 4. metric 계산
        avg_credibility = self._compute_average_credibility(validated_evidence)

        consistency_score = float(validation_result.get("evidence_consistency", 0.0))
        fact_overlap_score = float(validation_result.get("fact_overlap_score", 0.0))
        contradiction_score = float(validation_result.get("contradiction_score", 0.0))

        metrics = {
            "validated_count": len(validated_evidence),
            "conflicting_count": len(conflicting_evidence),
            "avg_credibility_score": avg_credibility,
            "evidence_consistency": consistency_score,
            "fact_overlap_score": fact_overlap_score,
            "contradiction_score": contradiction_score,
            "pass": (
                len(validated_evidence) > 0
                and avg_credibility >= VALIDATION_CREDIBILITY_THRESHOLD
                and consistency_score >= VALIDATION_CONSISTENCY_THRESHOLD
                and fact_overlap_score >= WEB_FACT_OVERLAP_THRESHOLD
                and contradiction_score >= WEB_CONTRADICTION_THRESHOLD
            ),
        }

        return {
            "validated_evidence": validated_evidence,
            "conflicting_evidence": conflicting_evidence,
            "metrics": metrics,
        }

    def _validate_with_llm(self, documents: List[Dict]) -> Dict:
        """
        LLM을 활용해 근거 검증 수행
        """

        doc_text = ""
        for idx, doc in enumerate(documents):
            doc_text += f"""
[{idx}]
title: {doc.get("title", "")}
content: {doc.get("content", "")[:500]}
source: {doc.get("url", "")}
"""

        system_prompt = """
You are a validation agent for semiconductor technology evidence review.

Your job:
- identify reliable evidence
- identify conflicting or weak evidence
- assign a credibility_score (0.0 ~ 1.0) to each validated evidence item
- evaluate evidence consistency
- estimate fact overlap across sources
- estimate contradiction score

Do not force consistency.
If there are differences in maturity claims, roadmap timing, production readiness, or technology feasibility, include them as conflicting evidence.

Return JSON only in this schema:
{
  "validated_evidence": [
    {
      "title": "...",
      "content": "...",
      "url": "...",
      "source_type": "...",
      "credibility_score": 0.0
    }
  ],
  "conflicting_evidence": [
    {
      "title": "...",
      "content": "...",
      "url": "...",
      "source_type": "...",
      "credibility_score": 0.0
    }
  ],
  "evidence_consistency": 0.0,
  "fact_overlap_score": 0.0,
  "contradiction_score": 0.0
}
"""

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": doc_text},
            ],
        )

        import json

        content = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(content)
        except Exception:
            return {
                "validated_evidence": [],
                "conflicting_evidence": [],
                "evidence_consistency": 0.0,
                "fact_overlap_score": 0.0,
                "contradiction_score": 0.0,
            }

        return parsed

    def _compute_average_credibility(self, evidence: List[Dict]) -> float:
        if not evidence:
            return 0.0

        scores = [float(doc.get("credibility_score", 0)) for doc in evidence]

        return sum(scores) / len(scores)

    def _empty_metrics(self) -> Dict:
        return {
            "validated_count": 0,
            "conflicting_count": 0,
            "avg_credibility_score": 0.0,
            "evidence_consistency": 0.0,
            "fact_overlap_score": 0.0,
            "contradiction_score": 0.0,
            "pass": False,
        }
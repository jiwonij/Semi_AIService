import json
import re
from typing import Dict, List

from openai import OpenAI

from utils.config import OPENAI_API_KEY, QUERY_COVERAGE_THRESHOLD, QUERY_MIN_COUNT


class QueryPlanningService:
    """
    사용자 질의를 바탕으로 분석 대상 기업, 기술, 검색 쿼리를 구조화하는 서비스.
    생성은 LLM이 수행하되, 최종 비교축은 코드에서 안정적으로 보정한다.
    """

    def __init__(self) -> None:
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.min_query_count = QUERY_MIN_COUNT
        self.coverage_threshold = QUERY_COVERAGE_THRESHOLD

        self.fixed_companies = ["SK hynix", "Samsung", "Micron"]
        self.fixed_technologies = ["HBM4", "PIM", "CXL", "Advanced Packaging"]

        self.technology_aliases = {
            "HBM4": [
                "hbm4",
                "hbm4e",
                "high bandwidth memory 4",
                "high bandwidth memory",
                "3d-stacked dram",
                "3d stacking",
                "stacked dram",
            ],
            "PIM": [
                "pim",
                "processing in memory",
                "processing-in-memory",
                "processing in memory (pim)",
                "in-memory computing",
                "near-memory computing",
                "memory-centric computing",
            ],
            "CXL": [
                "cxl",
                "compute express link",
                "compute express link (cxl)",
                "cxl memory",
                "cxl memory expansion",
                "memory interface technologies",
                "memory interface standards",
            ],
            "Advanced Packaging": [
                "advanced packaging",
                "advanced packaging for memory",
                "memory packaging",
                "advanced packaging for semiconductor memory",
                "2.5d packaging",
                "3d packaging",
                "tsv",
                "through silicon via",
                "interposer",
                "chiplet packaging",
            ],
        }

    def run(self, user_query: str) -> Dict:
        plan = self._generate_plan_with_llm(user_query)
        metrics = self._evaluate_plan(plan)

        return {
            "target_companies": plan.get("target_companies", []),
            "target_technologies": plan.get("target_technologies", []),
            "search_queries": plan.get("search_queries", []),
            "metrics": metrics,
        }

    def _generate_plan_with_llm(self, user_query: str) -> Dict:
        if not self.client:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

        system_prompt = """
You are a Query Planning Agent for a semiconductor technology strategy analysis workflow.

The analysis is centered on SK hynix.

Your job:
1. Identify relevant technologies for technology strategy analysis.
2. Generate search queries for retrieval and web search.

Requirements:
- Output must be valid JSON only.
- search_queries must contain at least 5 items.
- The goal is not a single-technology summary, but a comparative technology strategy analysis.
- Use SK hynix, Samsung, and Micron as the core comparison companies.
- Use HBM4, PIM, CXL, and Advanced Packaging as the core comparison technologies.
- Queries should support:
  - competitor comparison
  - TRL estimation
  - threat analysis
  - market/technology trend analysis
  - indirect evidence search such as patents, publications, hiring, investment, roadmap

Return JSON in this schema:
{
  "target_companies": ["SK hynix", "Samsung", "Micron"],
  "target_technologies": ["HBM4", "PIM", "CXL", "Advanced Packaging"],
  "search_queries": ["..."]
}
"""

        user_prompt = f"""
User query:
{user_query}
"""

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = (response.choices[0].message.content or "").strip()
        parsed = self._safe_parse_json(content)

        if not isinstance(parsed, dict):
            raise ValueError("LLM 출력이 dict 형식이 아닙니다.")

        raw_technologies = parsed.get("target_technologies", [])
        search_queries = parsed.get("search_queries", [])

        if not isinstance(raw_technologies, list):
            raw_technologies = []
        if not isinstance(search_queries, list):
            search_queries = []

        normalized_queries = [
            query.strip()
            for query in search_queries
            if isinstance(query, str) and query.strip()
        ]

        normalized_technologies = self._normalize_technologies(raw_technologies)

        if len(normalized_queries) < self.min_query_count:
            normalized_queries = self._build_fallback_queries(user_query)

        parsed["target_companies"] = self.fixed_companies
        parsed["target_technologies"] = normalized_technologies
        parsed["search_queries"] = normalized_queries

        return parsed

    def _normalize_technologies(self, raw_technologies: List[str]) -> List[str]:
        normalized = []

        for tech in raw_technologies:
            if not isinstance(tech, str) or not tech.strip():
                continue

            lowered = tech.lower().strip()
            matched_label = None

            for canonical, aliases in self.technology_aliases.items():
                if lowered == canonical.lower() or lowered in aliases:
                    matched_label = canonical
                    break

                for alias in aliases:
                    if alias in lowered:
                        matched_label = canonical
                        break

                if matched_label:
                    break

            if matched_label and matched_label not in normalized:
                normalized.append(matched_label)

        for canonical in self.fixed_technologies:
            if canonical not in normalized:
                normalized.append(canonical)

        return normalized

    def _build_fallback_queries(self, user_query: str) -> List[str]:
        queries = [
            f"{user_query}",
            "SK hynix Samsung Micron HBM4 HBM4E technology comparison",
            "SK hynix Samsung Micron PIM Processing In Memory technology comparison",
            "SK hynix Samsung Micron CXL Compute Express Link technology comparison",
            "SK hynix Samsung Micron Advanced Packaging TSV Interposer comparison",
            "HBM4 HBM4E TRL patent roadmap production comparison",
            "PIM Processing In Memory TRL patent hiring publication comparison",
            "CXL Compute Express Link TRL roadmap ecosystem investment comparison",
            "Advanced Packaging TSV Interposer roadmap comparison",
            "SK hynix semiconductor R&D threat analysis HBM4 PIM CXL Advanced Packaging",
        ]

        deduplicated_queries = []
        seen = set()

        for query in queries:
            normalized = query.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                deduplicated_queries.append(query)

        return deduplicated_queries

    def _safe_parse_json(self, content: str) -> Dict:
        if not content:
            raise ValueError("LLM 응답이 비어 있습니다.")

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if fenced_match:
            return json.loads(fenced_match.group(1))

        brace_match = re.search(r"(\{.*\})", content, re.DOTALL)
        if brace_match:
            return json.loads(brace_match.group(1))

        raise ValueError(f"JSON 파싱 실패. LLM 응답 원문: {content}")

    def _evaluate_plan(self, plan: Dict) -> Dict:
        target_companies = plan.get("target_companies", [])
        target_technologies = plan.get("target_technologies", [])
        search_queries = plan.get("search_queries", [])

        coverage_score = self._compute_query_coverage(
            queries=search_queries,
            companies=target_companies,
            technologies=target_technologies,
        )

        return {
            "query_count": len(search_queries),
            "query_coverage_score": coverage_score,
            "pass": (
                len(search_queries) >= self.min_query_count
                and coverage_score >= self.coverage_threshold
            ),
        }

    def _compute_query_coverage(
        self,
        queries: List[str],
        companies: List[str],
        technologies: List[str],
    ) -> float:
        if not queries:
            return 0.0

        keywords = [
            keyword.lower().strip()
            for keyword in (companies + technologies)
            if isinstance(keyword, str) and keyword.strip()
        ]

        if not keywords:
            return 0.0

        covered_count = 0
        for query in queries:
            lowered_query = query.lower()
            if any(keyword in lowered_query for keyword in keywords):
                covered_count += 1

        return covered_count / len(queries)
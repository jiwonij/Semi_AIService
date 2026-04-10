import os
import re
from pathlib import Path
from typing import Dict, List

from openai import OpenAI
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from utils.config import (
    OPENAI_API_KEY,
    REPORT_COMPLETENESS_THRESHOLD,
    REPORT_DIR,
    REPORT_MIN_CITATION_COUNT,
)


class ReportService:
    """
    최종 기술 전략 분석 보고서를 생성하고 PDF로 저장하는 서비스.
    """

    def __init__(self) -> None:
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.font_name = self._register_korean_font()
        self.title_style, self.body_style, self.note_style = self._build_styles()

    def run(
        self,
        user_query: str,
        analysis_result: Dict,
        trl_result: Dict,
        validated_evidence: List[Dict],
    ) -> Dict:
        if not self.client:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

        report_text = self._generate_report_with_llm(
            user_query=user_query,
            analysis_result=analysis_result,
            trl_result=trl_result,
            validated_evidence=validated_evidence,
        )

        report_text = self._normalize_report_headers(report_text)

        metrics = self._evaluate_report(report_text, validated_evidence)
        pdf_path = self._save_pdf(report_text, trl_result)

        return {
            "report_draft": report_text,
            "final_report": pdf_path,
            "metrics": metrics,
        }

    def _register_korean_font(self) -> str:
        font_candidates = [
            ("MalgunGothic", Path("C:/Windows/Fonts/malgun.ttf")),
            ("MalgunGothicBold", Path("C:/Windows/Fonts/malgunbd.ttf")),
            ("Gulim", Path("C:/Windows/Fonts/gulim.ttc")),
            ("Batang", Path("C:/Windows/Fonts/batang.ttc")),
        ]

        for font_name, font_path in font_candidates:
            if font_path.exists():
                pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
                return font_name

        raise FileNotFoundError(
            "사용 가능한 한글 폰트를 찾지 못했습니다. "
            "C:/Windows/Fonts 아래의 malgun.ttf 또는 gulim.ttc를 확인하세요."
        )

    def _build_styles(self):
        title_style = ParagraphStyle(
            name="ReportTitle",
            fontName=self.font_name,
            fontSize=15,
            leading=26,
            alignment=TA_LEFT,
            spaceAfter=10,
        )

        body_style = ParagraphStyle(
            name="ReportBody",
            fontName=self.font_name,
            fontSize=10.5,
            leading=20,
            alignment=TA_LEFT,
            wordWrap="CJK",
            spaceAfter=10,
        )

        note_style = ParagraphStyle(
            name="ReportNote",
            fontName=self.font_name,
            fontSize=9.5,
            leading=16,
            alignment=TA_LEFT,
            wordWrap="CJK",
            spaceAfter=8,
        )

        return title_style, body_style, note_style

    def _generate_report_with_llm(
        self,
        user_query: str,
        analysis_result: Dict,
        trl_result: Dict,
        validated_evidence: List[Dict],
    ) -> str:
        evidence_text = self._format_references(validated_evidence)
        trl_summary_text = "\n".join(self._build_trl_paragraphs(trl_result))

        system_prompt = """
당신은 반도체 기술 전략 보고서를 작성하는 분석가다.

반드시 아래 규칙을 지켜라.
- 보고서 전체를 한국어로만 작성한다.
- 기술명, 회사명, 제품명, 규격명 같은 고유명사를 제외하면 설명 문장은 한국어로 작성한다.
- 보고서는 SK hynix 중심의 기술 전략 보고서여야 한다.
- 경쟁사는 Samsung, Micron 기준으로 비교한다.
- 기술 비교, 기술 성숙도, 전략적 시사점을 포함한다.
- 요약은 충분히 자세하게 작성한다.
- 문장은 보고서 문체로 자연스럽게 작성한다.
- 불필요한 영어 헤더를 쓰지 않는다.
- TRL 4~6 구간은 공개 정보만으로 직접 검증하기 어려운 추정 영역임을 반드시 명시한다.
- TRL 4~6 구간은 특허, 논문, 학회, 채용, 투자, 로드맵 등 간접 지표를 기반으로 해석했음을 반드시 명시한다.

반드시 아래 구조를 따른다.

제목
요약
1. 분석 배경
2. 분석 대상 기술 현황
3. 경쟁사 동향 분석
4. 전략적 시사점
참고문헌
""".strip()

        user_prompt = f"""
사용자 질의:
{user_query}

분석 결과:
{analysis_result}

TRL 결과:
{trl_result}

TRL 비교 요약:
{trl_summary_text}

참고 자료:
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

        return response.choices[0].message.content.strip()

    def _normalize_report_headers(self, report_text: str) -> str:
        text = report_text.strip()

        replacements = {
            "TITLE": "제목",
            "SUMMARY": "요약",
            "REFERENCE": "참고문헌",
            "REFERENCES": "참고문헌",
        }

        for src, dst in replacements.items():
            text = re.sub(rf"^{src}$", dst, text, flags=re.MULTILINE)

        return text

    def _evaluate_report(self, report_text: str, validated_evidence: List[Dict]) -> Dict:
        required_sections = [
            "요약",
            "1. 분석 배경",
            "2. 분석 대상 기술 현황",
            "3. 경쟁사 동향 분석",
            "4. 전략적 시사점",
            "참고문헌",
        ]

        section_hits = sum(1 for section in required_sections if section in report_text)
        completeness = round(section_hits / len(required_sections), 4)
        citation_count = min(len(validated_evidence), 10)

        return {
            "citation_count": citation_count,
            "section_completeness": completeness,
            "pass": (
                citation_count >= REPORT_MIN_CITATION_COUNT
                and completeness >= REPORT_COMPLETENESS_THRESHOLD
            ),
        }

    def _save_pdf(self, report_text: str, trl_result: Dict) -> str:
        os.makedirs(REPORT_DIR, exist_ok=True)
        file_path = os.path.join(REPORT_DIR, "technology_strategy_report.pdf")

        doc = SimpleDocTemplate(
            file_path,
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
        )

        content = []

        for line in report_text.split("\n"):
            safe_line = line.strip()

            if not safe_line:
                content.append(Spacer(1, 10))
                continue

            if safe_line.startswith("참고문헌"):
                content.append(Spacer(1, 16))
                content.append(Paragraph("TRL 비교 요약", self.title_style))
                content.append(Spacer(1, 8))

                for paragraph in self._build_trl_paragraphs(trl_result):
                    content.append(Paragraph(paragraph, self.body_style))
                    content.append(Spacer(1, 6))

                note_text = (
                    "주: TRL 4~6 구간은 공개 정보만으로 직접 검증하기 어려운 추정 영역이며, "
                    "특허 출원 패턴, 학회·논문 활동, 채용 키워드, 투자 신호, 로드맵 발표 등 "
                    "간접 지표를 바탕으로 판단했다."
                )
                content.append(Paragraph(note_text, self.note_style))
                content.append(Spacer(1, 12))

            if safe_line in {"제목", "요약", "참고문헌"}:
                content.append(Paragraph(safe_line, self.title_style))
            elif safe_line.startswith(("1. ", "2. ", "3. ", "4. ")):
                content.append(Spacer(1, 12))
                content.append(Paragraph(safe_line, self.title_style))
            else:
                content.append(Paragraph(safe_line, self.body_style))

            content.append(Spacer(1, 6))

        doc.build(content)
        return file_path

    def _build_trl_paragraphs(self, trl_result: Dict) -> List[str]:
        paragraphs: List[str] = []
        trl_comparison = trl_result.get("trl_comparison", {})

        if not trl_comparison:
            return ["TRL 비교 결과를 생성할 수 있는 충분한 정보가 확보되지 않았다."]

        for technology, data in trl_comparison.items():
            sk = data.get("SK hynix", {})
            samsung = data.get("Samsung", {})
            micron = data.get("Micron", {})

            def fmt_company(name: str, company_data: Dict) -> str:
                level = company_data.get("trl_level", "-")
                conf = float(company_data.get("trl_confidence", 0.0))
                assessment = self._to_korean_assessment(company_data.get("assessment", ""))
                signals = company_data.get("indirect_signals_used", [])
                citations = company_data.get("citations", [])

                signal_text = ""
                if signals:
                    signal_text = f" 주요 신호로는 {', '.join(map(str, signals[:3]))} 등이 확인된다."

                citation_text = ""
                if citations:
                    citation_text = f" 근거 자료는 {', '.join(map(str, citations[:2]))} 등을 참고했다."

                return (
                    f"{name}는 TRL {level} "
                    f"(신뢰도 {conf:.2f}) 수준으로 평가되며, "
                    f"{assessment}{signal_text}{citation_text}"
                )

            summary = self._to_korean_summary(str(data.get("comparison_summary", "")).strip())

            paragraph = (
                f"{technology} 기준으로 보면, "
                f"{fmt_company('SK hynix', sk)} "
                f"{fmt_company('Samsung', samsung)} "
                f"{fmt_company('Micron', micron)} "
                f"종합적으로는 {summary}"
            )

            paragraphs.append(paragraph)

        return paragraphs

    def _to_korean_assessment(self, text: str) -> str:
        clean = str(text).strip()

        if not clean:
            return "공개 자료 기준으로 기술 성숙도를 판단할 수 있는 근거가 제한적이다."

        if re.search(r"[A-Za-z]{4,}", clean):
            return self._translate_to_korean(clean)

        return clean

    def _to_korean_summary(self, text: str) -> str:
        clean = str(text).strip()

        if not clean:
            return "공개 근거상 뚜렷한 우위를 단정하기 어렵다."

        if re.search(r"[A-Za-z]{4,}", clean):
            return self._translate_to_korean(clean)

        return clean

    def _translate_to_korean(self, text: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "다음 문장을 반도체 기술 전략 보고서 문체의 자연스러운 한국어로 바꿔라. "
                            "직역하지 말고 의미를 유지하면서 매끄럽게 정리하라. "
                            "회사명, 기술명, 제품명은 원문 표기를 유지할 수 있다."
                        ),
                    },
                    {
                        "role": "user",
                        "content": text[:1500],
                    },
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "기술 성숙도 판단을 위한 근거가 일부 확인되나 추가 검증이 필요하다."

    def _format_references(self, validated_evidence: List[Dict]) -> str:
        lines = []
        for idx, doc in enumerate(validated_evidence[:10], start=1):
            lines.append(
                f"[{idx}] title={doc.get('title', '')} | "
                f"source_type={doc.get('source_type', '')} | "
                f"url={doc.get('url', '')}"
            )
        return "\n".join(lines)
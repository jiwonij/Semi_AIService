import os
from typing import Dict, List

from openai import OpenAI
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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
        self.title_style, self.body_style, self.table_note_style = self._build_styles()

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

        metrics = self._evaluate_report(report_text, validated_evidence)
        pdf_path = self._save_pdf(report_text, trl_result)

        return {
            "report_draft": report_text,
            "final_report": pdf_path,
            "metrics": metrics,
        }

    def _register_korean_font(self) -> str:
        font_candidates = [
            ("AppleGothic", "/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
            ("ArialUnicodeMS", "/Library/Fonts/Arial Unicode.ttf"),
        ]

        for font_name, font_path in font_candidates:
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                return font_name

        raise FileNotFoundError("사용 가능한 한글 폰트를 찾지 못했습니다.")

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

        table_note_style = ParagraphStyle(
            name="TableNote",
            fontName=self.font_name,
            fontSize=9.5,
            leading=16,
            alignment=TA_LEFT,
            wordWrap="CJK",
            spaceAfter=8,
        )

        return title_style, body_style, table_note_style

    def _generate_report_with_llm(
        self,
        user_query: str,
        analysis_result: Dict,
        trl_result: Dict,
        validated_evidence: List[Dict],
    ) -> str:
        evidence_text = self._format_references(validated_evidence)

        system_prompt = """
You are an LLM Report Agent for semiconductor technology strategy reporting.

Write the report in Korean.
The report must be centered on SK hynix and its competitors.
The report must focus on technology comparison, technology maturity, and threat level across multiple technology axes.

Use the exact structure below:

제목 : SK hynix 기술 전략 분석 보고서
SUMMARY
1. 분석 배경
2. 분석 대상 기술 현황
3. 경쟁사 동향 분석
4. 전략적 시사점
REFERENCE

Requirements:
- SUMMARY must be substantial and close to half-page length.
- Each main section must contain detailed explanation, not short bullets only.
- Section 2 must explain each technology axis in detail, including technical meaning, importance, implementation direction, and current industry movement.
- Section 3 must compare SK hynix, Samsung, and Micron in detail for each major technology axis.
- The report must explicitly compare competitor-specific TRL and threat level.
- The report must explicitly state that TRL 4~6 is an estimation zone.
- The report must explicitly acknowledge that TRL 4~6 cannot be directly verified using only public information.
- When discussing TRL 4~6, explain that the estimate is based on indirect indicators such as patent filing patterns, publication and conference activity changes, hiring keywords, investment signals, and roadmap announcements.
- Write enough detail for an R&D strategy reader.
- Do not make the report too short.
- Prefer paragraph-style explanation with supporting detail rather than only compact summary lines.
"""

        user_prompt = f"""
User query:
{user_query}

Analysis result:
{analysis_result}

TRL result:
{trl_result}

Evidence references:
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

    def _evaluate_report(self, report_text: str, validated_evidence: List[Dict]) -> Dict:
        required_sections = [
            "SUMMARY",
            "1. 분석 배경",
            "2. 분석 대상 기술 현황",
            "3. 경쟁사 동향 분석",
            "4. 전략적 시사점",
            "REFERENCE",
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

            # 🔥 REFERENCE 만나기 직전에 TRL 삽입
            if safe_line.startswith("REFERENCE"):
                content.append(Spacer(1, 18))
                content.append(Paragraph("TRL 비교표", self.title_style))
                content.append(Spacer(1, 10))

                trl_table_data = self._build_trl_table(trl_result)

                table = Table(
                    trl_table_data,
                    colWidths=[35 * mm, 30 * mm, 30 * mm, 30 * mm, 55 * mm],
                    repeatRows=1,
                )

                table.setStyle(
                    TableStyle([
                        ("FONTNAME", (0, 0), (-1, -1), self.font_name),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ])
                )

                content.append(table)
                content.append(Spacer(1, 10))

            # 기존 텍스트 추가
            if safe_line == "SUMMARY":
                content.append(Paragraph(safe_line, self.title_style))
            elif safe_line.startswith(("1. ", "2. ", "3. ", "4. ")):
                content.append(Paragraph(safe_line, self.title_style))
            else:
                content.append(Paragraph(safe_line, self.body_style))

            content.append(Spacer(1, 6))

        content.append(Spacer(1, 18))
        content.append(Paragraph("TRL 비교표", self.title_style))
        content.append(Spacer(1, 10))

        trl_table_data = self._build_trl_table(trl_result)

        table = Table(
            trl_table_data,
            colWidths=[35 * mm, 30 * mm, 30 * mm, 30 * mm, 55 * mm],
            repeatRows=1,
        )

        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), self.font_name),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("LEADING", (0, 0), (-1, -1), 12),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("ALIGN", (4, 1), (4, -1), "LEFT"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        content.append(table)
        content.append(Spacer(1, 10))

        note_text = (
            "주: TRL 4~6 구간은 공개 정보만으로 직접 검증하기 어려운 추정 영역이며, "
            "특허 출원 패턴, 학회·논문 활동, 채용 키워드, 투자 신호, 로드맵 발표 등 "
            "간접 지표를 바탕으로 판단함."
        )
        content.append(Paragraph(note_text, self.table_note_style))

        doc.build(content)
        return file_path

    def _build_trl_table(self, trl_result: Dict) -> List[List[str]]:
        table_data = [["기술", "SK hynix", "Samsung", "Micron", "비고"]]

        trl_comparison = trl_result.get("trl_comparison", {})

        if not trl_comparison:
            table_data.append(["데이터 없음", "-", "-", "-", "TRL 비교 결과 없음"])
            return table_data

        for tech, data in trl_comparison.items():
            sk = data.get("SK hynix", {})
            samsung = data.get("Samsung", {})
            micron = data.get("Micron", {})

            def fmt(d: Dict) -> str:
                level = d.get("trl_level", "-")
                conf = d.get("trl_confidence", 0)
                try:
                    conf_text = f"{float(conf):.2f}"
                except Exception:
                    conf_text = "0.00"
                return f"TRL {level}\n(conf. {conf_text})"

            row = [
                tech,
                fmt(sk),
                fmt(samsung),
                fmt(micron),
                data.get("comparison_summary", "")[:80],
            ]

            table_data.append(row)

        return table_data

    def _format_references(self, validated_evidence: List[Dict]) -> str:
        lines = []
        for idx, doc in enumerate(validated_evidence[:10], start=1):
            lines.append(
                f"[{idx}] title={doc.get('title', '')} | "
                f"source_type={doc.get('source_type', '')} | "
                f"url={doc.get('url', '')}"
            )
        return "\n".join(lines)
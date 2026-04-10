## Semiconductor Technology Strategy AI Service

### Overview

반도체 메모리 기술 경쟁력 및 시장 기회 분석을 위한 Multi-Agent 기반 기술 전략 리포트 생성 시스템

---

## Objective

PDF 자료와 웹 검색 결과를 함께 활용하여 SK hynix 중심의 차세대 메모리 기술 전략, 경쟁사 동향, 기술 성숙도(TRL), 위협 요인을 분석하고 PDF 보고서를 생성한다.

---

### Features

* PDF 기반 문서 → 벡터화 (FAISS)
* RAG 기반 근거 검색
* 웹 검색 + 크롤링 기반 외부 정보 확장
* Validation 단계에서 근거 신뢰도/일관성 검증
* TRL 기반 기술 성숙도 평가
* 자동 기술 전략 보고서 생성 (PDF)

---

### Web Search & Bias Mitigation

* 다양한 쿼리(Query Planning)로 검색 편향 최소화
* 서로 다른 도메인에서 데이터 수집 (domain diversity)
* Retrieval + Web 결과 통합 후 Validation 수행
* 충돌 정보(conflicting evidence) 분리 및 비교
* 근거 일관성(evidence consistency) 기반 신뢰도 평가

---

### Tech Stack

* LangGraph, LangChain
* GPT-4.1-mini
* FAISS (Hit@5, MRR)
* BAAI/bge-m3

---

### Architecture

Query Planning → Retrieval → Web Search → Validation → Analysis → TRL → Report

---

### Directory Structure

```text
.
├── data/
│   ├── raw/                  # 원본 PDF 문서
│   └── vectorstore/          # FAISS 인덱스 저장 경로
├── eval/
│   ├── qa_retrieval_eval_dataset.json   # Retrieval 평가용 QA 데이터셋
│   ├── retrieval_eval.py               # Hit@K, MRR 계산 스크립트
│   └── retrieval_eval_result.xlsx      # 평가 결과 저장 파일
├── graph/
│   ├── graph_builder.py      # LangGraph 워크플로우 구성
│   ├── nodes.py              # 각 Agent 노드 정의
│   └── state.py              # Graph 상태 스키마 정의
├── outputs/
│   └── reports/              # 생성된 PDF 보고서 저장
├── services/
│   ├── analysis_service.py           # 기술 및 시장 분석 로직
│   ├── query_planning_service.py     # 검색 쿼리 생성 로직
│   ├── report_service.py             # PDF 보고서 생성
│   ├── retrieval_service.py          # 벡터 검색 및 문서 조회
│   ├── trl_service.py                # TRL 평가 로직
│   ├── validation_service.py         # 근거 신뢰도 및 일관성 검증
│   └── web_search_service.py         # 웹 검색 및 크롤링
├── utils/
│   └── config.py             # 환경 변수 및 경로 설정
├── build_index.py            # FAISS 인덱스 생성 스크립트
├── main.py                   # 전체 파이프라인 실행
├── requirements.txt
└── README.md
```

---

### Retrieval Evaluation

Hit@5: 0.95
MRR: 0.8667

---

### Run

```bash
python build_index.py
python main.py
```

---

### Pipeline Metrics (Summary)
- Query Planning, Retrieval, Validation, Analysis, TRL, Report 단계 모두 통과

---

### Output

outputs/reports/technology_strategy_report.pdf

---

### Contributor

Jiwon

* Agent Design
* Prompt Engineering
* Retrieval / Evaluation
* Web Search & Validation 설계
import json
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_PATH = BASE_DIR / "data" / "vectorstore"
EVAL_DATASET_PATH = BASE_DIR / "eval" / "qa_retrieval_eval_dataset.json"
OUTPUT_XLSX_PATH = BASE_DIR / "eval" / "retrieval_eval_result.xlsx"


def load_eval_dataset(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_hit_rate_at_k(results, relevant_docs, k: int) -> int:
    for doc in results[:k]:
        source = str(doc.metadata.get("source", ""))
        if any(rel in source for rel in relevant_docs):
            return 1
    return 0


def compute_mrr(results, relevant_docs) -> float:
    for rank, doc in enumerate(results, start=1):
        source = str(doc.metadata.get("source", ""))
        if any(rel in source for rel in relevant_docs):
            return 1.0 / rank
    return 0.0


def format_retrieved_docs(results, k: int) -> str:
    rows = []

    for rank, doc in enumerate(results[:k], start=1):
        source = str(doc.metadata.get("source", ""))
        content = doc.page_content.replace("\n", " ").strip()
        preview = content[:200]
        rows.append(f"[{rank}] {source} | {preview}")

    return "\n".join(rows)


def evaluate_retrieval(
    vectorstore,
    eval_data: List[Dict],
    k: int = 5
) -> Tuple[Dict[str, float], List[Dict]]:
    if not eval_data:
        return {f"Hit@{k}": 0.0, "MRR": 0.0}, []

    hit_total = 0
    mrr_total = 0
    rows = []

    for item in eval_data:
        query = item["query"]
        ground_truth_answer = item.get("answer", "")
        question_type = item.get("question_type", "")
        relevant_docs = item.get("relevant_docs", [])

        results = vectorstore.similarity_search(query, k=k)

        hit = compute_hit_rate_at_k(results, relevant_docs, k)
        mrr = compute_mrr(results, relevant_docs)

        hit_total += hit
        mrr_total += mrr

        top1_source = ""
        top1_content = ""

        if results:
            top1_source = str(results[0].metadata.get("source", ""))
            top1_content = results[0].page_content.replace("\n", " ").strip()[:300]

        rows.append({
            "id": item.get("id", ""),
            "question_type": question_type,
            "query": query,
            "ground_truth_answer": ground_truth_answer,
            "relevant_docs": " | ".join(relevant_docs),
            "top1_source": top1_source,
            "top1_content": top1_content,
            "retrieved_docs_topk": format_retrieved_docs(results, k),
            "hit@5": hit,
            "mrr": round(mrr, 4),
        })

    summary = {
        f"Hit@{k}": round(hit_total / len(eval_data), 4),
        "MRR": round(mrr_total / len(eval_data), 4),
    }

    return summary, rows


if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    eval_data = load_eval_dataset(EVAL_DATASET_PATH)

    summary_metrics, detailed_rows = evaluate_retrieval(
        vectorstore=vectorstore,
        eval_data=eval_data,
        k=5
    )

    print("=== Retrieval Evaluation ===")
    print(summary_metrics)

    df = pd.DataFrame(detailed_rows, columns=[
        "id",
        "query",
        "relevant_docs",
        "top1_source",
        "hit@5",
        "mrr",
    ])

    df.to_excel(OUTPUT_XLSX_PATH, index=False)

    print(f"상세 결과 저장 완료: {OUTPUT_XLSX_PATH}")
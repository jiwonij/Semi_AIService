from pathlib import Path
from typing import Dict, List, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import EMBEDDING_CANDIDATES, RAW_DATA_DIR, VECTORSTORE_DIR


class RetrievalService:
    """
    로컬 문서를 불러와 임베딩 기반 검색을 수행하는 서비스.
    벡터스토어가 저장되어 있으면 불러오고, 없으면 새로 생성 후 저장한다.
    성능 평가는 별도 evaluation 단계에서 수행한다.
    """

    def __init__(self) -> None:
        self.embedding_model_name = EMBEDDING_CANDIDATES[0]
        self.chunk_size = 900
        self.chunk_overlap = 150
        self.top_k = 8

    def run(self, search_queries: List[str]) -> Dict:
        embeddings = self._get_embeddings()
        vectorstore = self._load_or_build_vectorstore(embeddings)
        retrieved_evidence = self._retrieve(vectorstore, search_queries)

        scores = [item["score"] for item in retrieved_evidence if isinstance(item.get("score"), float)]
        avg_similarity_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "retrieved_evidence": retrieved_evidence,
            "metrics": {
                "retrieved_count": len(retrieved_evidence),
                "avg_similarity_score": round(avg_similarity_score, 4),
                "selected_embedding_model": self.embedding_model_name,
                "pass": len(retrieved_evidence) > 0,
            },
        }

    def build_and_save_index(self) -> Dict:
        embeddings = self._get_embeddings()
        documents = self._load_documents()

        if not documents:
            return {
                "document_count": 0,
                "chunk_count": 0,
                "saved_path": VECTORSTORE_DIR,
                "pass": False,
            }

        chunks = self._split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, embeddings)

        save_dir = Path(VECTORSTORE_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(save_dir))

        return {
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "saved_path": str(save_dir),
            "pass": True,
        }

    def _load_or_build_vectorstore(self, embeddings: HuggingFaceEmbeddings) -> FAISS:
        save_dir = Path(VECTORSTORE_DIR)
        index_file = save_dir / "index.faiss"
        pkl_file = save_dir / "index.pkl"

        if index_file.exists() and pkl_file.exists():
            return FAISS.load_local(
                str(save_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )

        documents = self._load_documents()
        if not documents:
            raise ValueError("data/raw에 문서가 없습니다.")

        chunks = self._split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, embeddings)

        save_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(save_dir))

        return vectorstore

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _load_documents(self) -> List:
        raw_dir = Path(RAW_DATA_DIR)
        if not raw_dir.exists():
            return []

        documents = []

        for file_path in raw_dir.rglob("*"):
            if not file_path.is_file():
                continue

            suffix = file_path.suffix.lower()

            try:
                if suffix == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                    documents.extend(loader.load())
                elif suffix == ".txt":
                    loader = TextLoader(str(file_path), encoding="utf-8")
                    documents.extend(loader.load())
                elif suffix in {".md", ".markdown"}:
                    loader = UnstructuredMarkdownLoader(str(file_path))
                    documents.extend(loader.load())
            except Exception:
                continue

        return documents

    def _split_documents(self, documents: List) -> List:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_documents(documents)

    def _retrieve(self, vectorstore: FAISS, search_queries: List[str]) -> List[Dict]:
        retrieved = []
        seen_keys = set()

        for query in search_queries:
            docs_with_scores: List[Tuple] = vectorstore.similarity_search_with_relevance_scores(
                query,
                k=self.top_k,
            )

            for doc, score in docs_with_scores:
                key = (
                    doc.metadata.get("source", ""),
                    doc.metadata.get("page", ""),
                    doc.page_content[:120],
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                source_path = doc.metadata.get("source", "local_document")

                retrieved.append(
                    {
                        "query": query,
                        "title": Path(source_path).name,
                        "content": doc.page_content,
                        "source_type": "internal_document",
                        "source_name": source_path,
                        "url": "",
                        "domain": "local",
                        "published_at": "",
                        "score": float(score),
                        "metadata": doc.metadata,
                    }
                )

        return retrieved

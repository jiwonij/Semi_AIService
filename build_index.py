from services.retrieval_service import RetrievalService


def main():
    service = RetrievalService()
    result = service.build_and_save_index()

    print("인덱스 생성 결과")
    print("문서 수:", result["document_count"])
    print("청크 수:", result["chunk_count"])
    print("저장 경로:", result["saved_path"])
    print("성공 여부:", result["pass"])


if __name__ == "__main__":
    main()
from ingestion.loaders import load_document
from ingestion.ingest import ingest_document
from pathlib import Path

working_dir = Path(__file__).resolve().parent

def main():
    docs = ingest_document(file_path=rf"{working_dir}\data\sample.pdf", filename="sample.pdf")
    # print(len(docs), docs[0].metadata)
    # print(docs[0].page_content)

    with open(rf"{working_dir}\data\sample.txt", "rb") as f:
        data = f.read()

    docs = ingest_document(file_bytes=data, filename="sample.txt")
    # print(len(docs), docs[0].metadata)
    # print(docs[0].page_content)


if __name__ == "__main__":
    main()

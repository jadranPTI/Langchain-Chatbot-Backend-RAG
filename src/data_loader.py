from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader,  UnstructuredWordDocumentLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader



def load_all_documents(data_dir: str) -> List[Any]:

    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")

    documents = []

    files = (
        list(data_path.glob("**/*.pdf")) +
        list(data_path.glob("**/*.txt")) +
        list(data_path.glob("**/*.csv")) +
        list(data_path.glob("**/*.docx")) +
        list(data_path.glob("**/*.doc"))
    )

    print(f"[DEBUG] Found {len(files)} files: {[str(f) for f in files]}")

    for file in files:
        print(f"[DEBUG] Loading file: {file}")

        try:
            extension = file.suffix.lower()

            if extension == ".pdf":
                loader = PyPDFLoader(str(file))

            elif extension == ".txt":
                loader = TextLoader(str(file), encoding="utf-8")

            elif extension == ".csv":
                loader = CSVLoader(str(file))

            elif extension in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(str(file))

            else:
                print(f"[DEBUG] Unsupported file type: {file}")
                continue

            loaded = loader.load()

            print(
                f"[DEBUG] Loaded {len(loaded)} documents from {file}"
            )

            documents.extend(loaded)

        except Exception as e:
            print(f"[ERROR] Failed to load {file}: {e}")

    return documents

# def load_all_documents(data_dir:str) -> List[Any]:

#     data_path = Path(data_dir).resolve()
#     print(f"[DEBUG] Data path: {data_path}")
#     documents = []

#     files = (
#         list(data_path.glob('**/*.pdf'))+
#         list(data_path.glob('**/*.txt')) +
#         list(data_path.glob('**/*.csv')) +
#         list(data_path.glob('**/*.docx'))+
#         list(data_path.glob('**/*.doc'))
#     )
#     print(f"[DEBUG] Found {len(files)} PDF Files: {[str(f) for f in files]}")
#     for pdf_file in files:
#         print(f"[DEBUG] Loading PDF: {pdf_file}")
#         try:
#             loader = PyPDFLoader(str(pdf_file))
#             loaded = loader.load()
            
#             print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
#             documents.extend(loaded)

#         except Exception as e:
#             print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

#     return documents















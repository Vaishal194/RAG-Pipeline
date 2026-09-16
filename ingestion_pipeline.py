from langchain_community.document_loaders import DirectoryLoader, TextLoader,PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

def load_document(doc_path):
    if not os.path.exists(doc_path):
        raise FileNotFoundError
    loader=DirectoryLoader(
        path=doc_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding":"utf-8"}
    )

    document=loader.load()

    for i,doc in enumerate(document):
        print(f"Document:{i+1} ")
        print(f"Document Page Content: {doc.page_content[:100]}")
        print(f"Size: {len(doc.page_content)}Chars")
        print(f"MetaData: {doc.metadata}")
    return document

def load_pdf(doc_path):

    loader=PyPDFDirectoryLoader(
        path=doc_path,
        glob="*.pdf"
    )

    document=loader.load()

    for i ,doc in enumerate(document):
        print(f"Document: {i+1}")
        print(f"Document Content: {doc.page_content}")
        print(f"Meta Data: {doc.metadata}")
    return document

def split_documents(document,chunk_size=880,chunk_overlap=0):
    splitted_docs=CharacterTextSplitter(
        chunk_overlap=chunk_overlap,
        chunk_size=chunk_size
    )

    chunks=splitted_docs.split_documents(document)

    for i ,doc in enumerate(chunks):
        print(f"Chunk {i+1}")
        print(f"Page Content {doc.page_content[:50]}")
        print(f"Meta Data: {doc.metadata}")
    return chunks

def create_vector_embeddings(chunks,persist_directory="db/chroma_db"):
    embedding_model=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    db=Chroma.from_documents(
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"},
        documents=chunks
    )
    return db
def main():
    doc=load_document("docs")
    chunks=split_documents(doc)
    db=create_vector_embeddings(chunks)
if __name__=="__main__":
    main()
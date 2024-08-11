import os
import shutil
import openai
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
CHROMA_PATH = "chroma"
DATA_PATH = "Data/thy"

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

def determine_format(content):
    if content.strip().startswith("Soru:"):
        return "qa"
    elif ":" in content and not content.strip().startswith("Soru:"):
        return "dictionary"
    else:
        return "unknown"

def split_text(documents: list[Document]):
    qa_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\nSoru:", "\n\n"],
        length_function=len,
        add_start_index=True,
    )
    
    dict_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n"],
        length_function=len,
        add_start_index=True,
    )
    
    chunks = []
    for doc in documents:
        format_type = determine_format(doc.page_content)
        if format_type == "qa":
            split_chunks = qa_splitter.split_text(doc.page_content)
        elif format_type == "dictionary":
            split_chunks = dict_splitter.split_text(doc.page_content)
        else:
            split_chunks = RecursiveCharacterTextSplitter().split_text(doc.page_content)
        
        for chunk in split_chunks:
            chunks.append(Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "format": format_type,
                    "chunk_size": len(chunk)
                }
            ))
    
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def main():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

if __name__ == "__main__":
    main()
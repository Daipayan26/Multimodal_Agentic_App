#Importing necessary libraries
from langchain_community.document_loaders import PyMuPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface  import HuggingFaceEmbeddings 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from typing import List
from langchain_core.documents import Document
from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()

### Data Loading 
def load_pdf_files(pdf_path):
    print(f"Loading pdf files from {pdf_path}...")
    loader=DirectoryLoader(pdf_path,glob="**/*.pdf",
                           show_progress=True,
                           loader_cls=PyMuPDFLoader)
    documents=loader.load()
    print(f"Finished loading pdf files from {pdf_path}.")
    return documents

# Removing unnecessary metadata from the documents to save memory and focus on content and source.

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

# Text Splitting 
def split_documents(minimal_documents, chunk_size: int = 500, chunk_overlap: int = 100) :
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_documents(minimal_documents)
    return text_chunks

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)
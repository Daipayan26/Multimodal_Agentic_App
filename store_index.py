from dotenv import load_dotenv
from src.helper import load_pdf_files, filter_to_minimal_docs, split_documents
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from src.prompt import prompt

load_dotenv()

extracted_data = load_pdf_files("pdf_data")
filtered_data= filter_to_minimal_docs(extracted_data)
text_chunks = split_documents(filtered_data)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
pc=Pinecone()

index_name="multimodal-agentic-app"
if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=384,  # Dimension of the embeddings
        metric= "cosine",  # Cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)
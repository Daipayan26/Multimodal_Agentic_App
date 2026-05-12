from flask import Flask, request, jsonify,render_template
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from src.helper import format_docs
from src.prompt import *
import os



app = Flask(__name__)
load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_name="multimodal-agentic-app"
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
chatModel = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# Build the RAG chain with proper document formatting
rag_chain = (
    {
        "context": itemgetter("input") | retriever | format_docs,
        "input": itemgetter("input")
    }
    | prompt
    | chatModel
    | StrOutputParser()
)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["GET","POST"])
def chat():
    # Handle both GET and POST requests
    if request.method == "POST":
        msg = request.form.get("msg")
    else:
        msg = request.args.get("msg")
    
    # Validate input
    if not msg or msg.strip() == "":
        return jsonify({"error": "Please provide a message"}), 400
    
    print(f"User message: {msg}")
    response = rag_chain.invoke({"input": msg})
    print(f"Bot response: {response}")
    return str(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080,debug=True)
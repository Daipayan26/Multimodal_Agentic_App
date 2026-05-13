from flask import Flask, request, jsonify,render_template
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from src.helper import format_docs
from src.prompt import *
import os


# In-memory chat history storage
class InMemoryChatHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []
    
    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
    
    def clear(self) -> None:
        self.messages = []

# Store for session histories
session_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create a chat history for a session"""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatHistory()
    return session_store[session_id]


app = Flask(__name__)
load_dotenv()

# Initialize components with error handling
embeddings = None
docsearch = None
retriever = None
chatModel = None
rag_chain = None
runnable_with_history = None
initialization_attempted = False
initialization_error = None

def initialize_components():
    global embeddings, docsearch, retriever, chatModel, rag_chain, runnable_with_history, initialization_attempted, initialization_error
    
    initialization_attempted = True
    
    try:
        print("Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        print("Initializing Pinecone...")
        index_name="multimodal-agentic-app"
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
        
        print("Initializing ChatGoogleGenerativeAI...")
        chatModel = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        # Build the RAG chain with proper document formatting and history support
        rag_chain = (
            {
                "context": itemgetter("input") | retriever | format_docs,
                "input": itemgetter("input"),
                "history": itemgetter("history")
            }
            | prompt
            | chatModel
            | StrOutputParser()
        )

        runnable_with_history = RunnableWithMessageHistory(
            runnable=rag_chain,
            get_session_history=get_session_history,
            input_messages_key='input',
            history_messages_key='history'
        )
        print("All components initialized successfully!")
        return True
    except Exception as e:
        error_msg = f"ERROR during initialization: {e}"
        print(error_msg)
        initialization_error = error_msg
        return False

# Initialize on first request
@app.before_request
def init_on_first_request():
    global rag_chain, initialization_attempted
    if not initialization_attempted:
        initialize_components()


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    global rag_chain, initialization_attempted, initialization_error
    if not initialization_attempted:
        return jsonify({"status": "initializing"}), 202
    if rag_chain is None:
        return jsonify({"status": "failed", "error": initialization_error}), 500
    return jsonify({"status": "ready"}), 200

@app.route("/get", methods=["GET","POST"])
def chat():
    global rag_chain, runnable_with_history, initialization_error
    
    # Ensure components are initialized
    if rag_chain is None or runnable_with_history is None:
        error_msg = initialization_error or "System not initialized. Please try again."
        return jsonify({"error": error_msg}), 500
    
    # Handle both GET and POST requests
    if request.method == "POST":
        msg = request.form.get("msg")
    else:
        msg = request.args.get("msg")
    
    # Validate input
    if not msg or msg.strip() == "":
        return jsonify({"error": "Please provide a message"}), 400
    
    try:
        print(f"User message: {msg}")
        response = ""
        for chunk in runnable_with_history.stream({'input': msg}, config={"configurable": {"session_id": "default"}}):
            response += chunk
            print(f"Bot response: {chunk}")
        return str(response)
    except Exception as e:
        print(f"ERROR in chat: {e}")
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080,debug=True)
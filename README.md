# Food Nutrition Chatbot (RAG Application)

A conversational AI application designed to act as a food nutrition assistant. It leverages **Retrieval-Augmented Generation (RAG)** to answer food nutrition questions concisely by retrieving relevant context from ingested PDF documents. 

Built with **LangChain**, **Flask**, **Pinecone**, and **Google Gemini Generative AI**.

## 🌟 Features

- **Document Ingestion**: Seamlessly load and split food nutrition PDF documents from the local directory (`pdf_data/`).
- **Vector Search Engine**: Uses **Pinecone** Serverless vector database for fast and accurate similarity search.
- **Advanced Embeddings**: Utilizes HuggingFace's `all-MiniLM-L6-v2` for generating high-quality text embeddings.
- **Generative AI Responses**: Integrates **Google Gemini 2.5 Flash** for highly contextual, intelligent answering.
- **Web Interface**: A clean and simple web UI built with **Flask** to interact with the assistant.

## 🛠️ Tech Stack

- **Framework**: Flask
- **Orchestration**: LangChain
- **LLM**: Google Gemini 2.5 Flash (`langchain-google-genai`)
- **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Store**: Pinecone
- **Package Management**: `uv` / `pip`

## 📁 Project Structure

```text
├── pdf_data/           # Put your PDF files here (e.g., food_nutrition.pdf)
├── src/
│   ├── helper.py       # Functions for PDF loading, filtering, and text splitting
│   └── prompt.py       # LangChain system prompt definitions
├── static/             # Static files (CSS, JS)
├── templates/          # HTML templates for the Flask app
├── app.py              # Main Flask web server application
├── store_index.py      # Script to ingest PDFs and index them in Pinecone
├── pyproject.toml      # Project dependencies and configuration
└── requirements.txt    # requirements file
```

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.12+ installed. 

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd Multimodal_Agentic_App
```

### 3. Install Dependencies
You can install dependencies using `pip`:
```bash
pip install -r requirements.txt
```
*(Alternatively, you can use `uv` as configured in `pyproject.toml`)*

### 4. Environment Variables
Create a `.env` file in the root directory and add the following API keys:
```ini
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_gemini_api_key
```

### 5. Ingest Data to Vector Database
Place your source PDF files inside the `pdf_data/` directory, then run the indexing script to embed and store them in Pinecone:
```bash
python store_index.py
```

### 6. Run the Application
Start the Flask web server:
```bash
python app.py
```
The application will be accessible at `http://localhost:8080`.

## 💡 Usage
1. Open the application in your web browser.
2. Type in your food nutrition question in the chat interface.
3. The RAG system will query Pinecone for the most relevant context and the LLM will generate a concise answer based on that specific context.

# main.py (Final Version with Dynamic File Upload)

# --- 1. Imports ---
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import faiss
import sqlite3
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from gradio_client import Client
import os
from contextlib import asynccontextmanager

# We need to import the functions we built in previous phases
from parser import parse_ufdr
from database_builder import build_database

# --- 2. Pydantic Model for Data Validation ---
class QueryRequest(BaseModel):
    question: str

# --- 3. App Initialization and Lifespan Management ---

# This dictionary will hold our AI models and the loaded index
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs ONCE when the server starts up.
    # It pre-loads the heavy AI models so they are ready for requests.
    print("--- Application Startup ---")
    print("Loading core AI models...")
    ml_models["sentence_model"] = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    ml_models["llm_client"] = Client("ashish5077/UFDR_SLM_server")
    print("✅ Core models loaded successfully.")
    print("--- Server is now ready to accept file uploads. ---")
    yield
    # This code runs when the server shuts down
    print("--- Application Shutdown ---")
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

# --- Add CORS Middleware ---
origins = [
    "http://localhost:3000",
    # Remember to add your Vercel URL here when you deploy
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. API Endpoints ---

@app.post("/upload")
async def upload_and_process_report(file: UploadFile = File(...)):
    """
    Receives a UFDR file, saves it, parses it, and builds the vector database.
    This makes the report ready for querying.
    """
    try:
        # Save the uploaded file to a temporary location on the server
        temp_file_path = "temp_report.zip"
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"File '{file.filename}' uploaded. Starting processing...")
        # Use our existing functions to process the file
        ufdr_data = parse_ufdr(temp_file_path)
        build_database(ufdr_data)
        
        # Load the newly created FAISS index into memory
        ml_models["faiss_index"] = faiss.read_index('report_index.faiss')
        
        print("✅ Database built and loaded successfully from the new report.")
        return {"message": f"Successfully processed '{file.filename}' and it is ready for analysis."}
        
    except Exception as e:
        print(f"Error during upload: {e}")
        return {"error": str(e)}

@app.post("/query")
def query_index(request: QueryRequest):
    """
    Receives a question and queries the most recently uploaded report.
    """
    # Check if a report has been processed and its index is loaded
    if "faiss_index" not in ml_models:
        return {"error": "No report has been uploaded and processed yet. Please upload a file first."}
        
    try:
        # --- Retrieval ---
        question_vector = ml_models["sentence_model"].encode([request.question])
        k = 5
        distances, indices = ml_models["faiss_index"].search(question_vector, k)
        result_indices = indices[0]
        
        conn = sqlite3.connect('report_data.db')
        cursor = conn.cursor()
        placeholders = ','.join('?' for _ in result_indices)
        query = f'SELECT content FROM chunks WHERE id IN ({placeholders})'
        ids_to_fetch = [int(i) for i in result_indices]
        cursor.execute(query, ids_to_fetch)
        results = cursor.fetchall()
        conn.close()
        retrieved_chunks = [row[0] for row in results]

        # --- Augmentation (Analytical Prompt) ---
        context = "\n---\n".join(retrieved_chunks)
        prompt = f"""
        You are an intelligent forensic data analyst AI. Your mission is to provide concise and accurate answers based *strictly* on the provided CONTEXT.

        Your capabilities include:
        1.  **Direct Extraction:** If the user asks for specific information (like a name or address), extract it directly.
        2.  **Counting & Aggregation:** If the user asks "how many" or "total", count the relevant items in the context.
        3.  **Summarization:** If the user asks for a general overview, summarize the relevant points.

        RULES:
        - Base your answer ONLY on the provided CONTEXT.
        - Do not make assumptions or use outside knowledge.
        - If the CONTEXT does not contain the necessary information, and only in that case, respond with "Information not found in report."

        CONTEXT:
        ---
        {context}
        ---

        QUESTION:
        {request.question}

        ANSWER:
        """

        # --- Generation ---
        final_answer = ml_models["llm_client"].predict(
                        prompt,
                        api_name="/predict"
        )
        return {"answer": final_answer}

    except Exception as e:
        print(f"Error during query: {e}")
        return {"error": str(e)}

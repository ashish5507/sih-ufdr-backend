# main.py (Final, Memory-Optimized, Production-Ready Version)

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import faiss
import sqlite3
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from gradio_client import Client
import os
from contextlib import asynccontextmanager

from parser import parse_ufdr
from database_builder import build_database

class QueryRequest(BaseModel):
    question: str

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup process now does nothing, keeping memory usage minimal.
    print("--- Application Startup: Server is ready. Models will be loaded on first use. ---")
    yield
    print("--- Application Shutdown ---")
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# --- Add your Vercel URL to this list ---
origins = [
    "http://localhost:3000",
    "https://sih-ufdr-frontend-g650xgtjl-ashishs-projects-5ee0cda8.vercel.app", 
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _load_sentence_model():
    """Helper function to lazy load the embedding model."""
    if "sentence_model" not in ml_models:
        print("Loading sentence-transformer model for the first time...")
        ml_models["sentence_model"] = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        print("✅ Sentence-transformer model loaded.")

def _load_llm_client():
    """Helper function to lazy load the LLM client."""
    if "llm_client" not in ml_models:
        print("Connecting to LLM service for the first time...")
        ml_models["llm_client"] = Client("ashish5077/UFDR_SLM_server")
        print("✅ LLM client connected.")

@app.post("/upload")
async def upload_and_process_report(file: UploadFile = File(...)):
    try:
        _load_sentence_model() # Load model on first upload
        temp_file_path = "temp_report.zip"
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        ufdr_data = parse_ufdr(temp_file_path)
        build_database(ufdr_data, ml_models["sentence_model"]) # Pass the loaded model
        ml_models["faiss_index"] = faiss.read_index('report_index.faiss')
        
        return {"message": f"Successfully processed '{file.filename}'"}
    except Exception as e:
        print(f"!!!!!!!! UPLOAD FAILED: {e} !!!!!!!!")
        return {"error": str(e)}

@app.post("/query")
def query_index(request: QueryRequest):
    if "faiss_index" not in ml_models:
        return {"error": "No report has been uploaded yet."}
    
    try:
        # Load models on first query if they haven't been loaded by an upload
        _load_sentence_model()
        _load_llm_client()

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

        context = "\n---\n".join(retrieved_chunks)
        prompt = f"CONTEXT: --- {context} --- QUESTION: {request.question} --- Based ONLY on the context, provide a direct answer."
        final_answer = ml_models["llm_client"].predict(prompt, api_name="/predict")
        return {"answer": final_answer}
    except Exception as e:
        print(f"!!!!!!!! QUERY FAILED: {e} !!!!!!!!")
        return {"error": str(e)}
# main.py (Final Version with Lazy Initialization)

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
    # This code now ONLY loads the local models at startup.
    # It no longer connects to the external LLM.
    print("--- Application Startup ---")
    print("Loading local embedding model...")
    ml_models["sentence_model"] = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    print("✅ Local models loaded successfully.")
    print("--- Server is now ready to accept file uploads. ---")
    yield
    print("--- Application Shutdown ---")
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000",
    "https://sih-ufdr-frontend-g650xgtjl-ashishs-projects-5ee0cda8.vercel.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_and_process_report(file: UploadFile = File(...)):
    try:
        temp_file_path = "temp_report.zip"
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"File '{file.filename}' uploaded. Starting processing...")
        ufdr_data = parse_ufdr(temp_file_path)
        build_database(ufdr_data)
        
        ml_models["faiss_index"] = faiss.read_index('report_index.faiss')
        
        print("✅ Database built and loaded successfully from new report.")
        return {"message": f"Successfully processed '{file.filename}' and it is ready for analysis."}
    except Exception as e:
        print(f"!!!!!!!! UPLOAD FAILED: {e} !!!!!!!!")
        return {"error": str(e)}

@app.post("/query")
def query_index(request: QueryRequest):
    # --- THIS IS THE LAZY INITIALIZATION FIX ---
    # Check if the LLM client has been loaded. If not, load it now.
    # This only runs ONCE, on the very first query.
    if "llm_client" not in ml_models:
        print("First query received. Initializing connection to LLM...")
        try:
            ml_models["llm_client"] = Client("ashish5077/UFDR_SLM_server")
            print("✅ LLM client connected successfully.")
        except Exception as e:
            print(f"!!!!!!!! LLM CONNECTION FAILED: {e} !!!!!!!!")
            return {"error": "Could not connect to the language model service."}
    # --- END OF FIX ---
    
    if "faiss_index" not in ml_models:
        return {"error": "No report has been uploaded and processed yet. Please upload a file first."}
    
    try:
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
        prompt = f"""
        You are an intelligent forensic data analyst AI. Your mission is to provide concise and accurate answers based *strictly* on the provided CONTEXT.
        Your capabilities include counting, summarizing, and direct extraction.
        If the CONTEXT does not contain the necessary information, respond with "Information not found in report."
        CONTEXT: --- {context} ---
        QUESTION: {request.question}
        ANSWER:
        """

        final_answer = ml_models["llm_client"].predict(prompt, api_name="/predict")
        return {"answer": final_answer}
    except Exception as e:
        print(f"!!!!!!!! QUERY FAILED: {e} !!!!!!!!")
        return {"error": str(e)}
# database_builder.py (Final Version)
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import os

# The function now accepts 'model' as a second argument
def build_database(parsed_data, model):
    """
    Builds a FAISS index and a SQLite database from the parsed UFDR data.
    """
    print("Step 1: Preparing text data for embedding...")
    texts_to_embed = []
    for item in parsed_data:
        if item['type'] == 'chat':
            text = f"Chat from {item['sender']} at {item['timestamp']}: {item['content']}"
        elif item['type'] == 'call':
            text = f"Call log at {item['timestamp']}: {item['direction']} call with {item['number_or_contact']}"
        elif item['type'] == 'contact':
            text = f"Contact entry: Name is {item['name']}, Number is {item['number']}"
        else:
            text = json.dumps(item)
        texts_to_embed.append(text)
    
    print("Step 2: Generating vector embeddings...")
    # This line is removed: model = SentenceTransformer(...)
    # We now use the model that was passed into the function.
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    print(f"Generated {len(embeddings)} vectors.")

    print("Step 3: Building and saving FAISS index...")
    vector_dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    index.add(embeddings)
    faiss.write_index(index, 'report_index.faiss')

    print("Step 4: Creating and populating SQLite database...")
    db_file = 'report_data.db'
    if os.path.exists(db_file):
        os.remove(db_file)
        
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, content TEXT NOT NULL)''')
    cursor.executemany('INSERT INTO chunks (id, content) VALUES (?, ?)', enumerate(texts_to_embed))
    conn.commit()
    conn.close()
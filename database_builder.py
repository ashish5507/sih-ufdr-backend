# Import necessary libraries
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import os

# Import the parser function we created in Phase 1
from parser import parse_ufdr

def build_database(parsed_data):
    """
    Builds a FAISS index and a SQLite database from the parsed UFDR data.

    Args:
        parsed_data (list): The list of dictionaries from the parse_ufdr function.
    """
    
    # --- Step 1: Prepare the Text Data for Embedding ---
    # We need a list of strings for the embedding model. We'll format our
    # structured data into human-readable strings. This process is our
    # simple and effective "chunking" strategy.
    
    print("Step 1: Preparing text data for embedding...")
    texts_to_embed = []
    for item in parsed_data:
        # Create a single, descriptive string for each piece of data
        if item['type'] == 'chat':
            text = f"Chat from {item['sender']} at {item['timestamp']}: {item['content']}"
        elif item['type'] == 'call':
            text = f"Call log at {item['timestamp']}: {item['direction']} call with {item['number_or_contact']}"
        elif item['type'] == 'contact':
            text = f"Contact entry: Name is {item['name']}, Number is {item['number']}"
        else:
            text = json.dumps(item) # Fallback for any other data types
        texts_to_embed.append(text)
    
    # --- Step 2: Generate Vector Embeddings ---
    # This is where the AI model reads our text and converts it into
    # numerical vectors that capture its semantic meaning.
    
    print("Step 2: Loading embedding model and generating vectors...")
    # Load the small, efficient model we chose.
    # The first time you run this, it will download the model files.
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

    # The model.encode() function is the core of the process.
    # It takes our list of strings and outputs a NumPy array of vectors.
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    print(f"Generated {len(embeddings)} vectors.")

    # --- Step 3: Build and Save the FAISS Index ---
    # FAISS is a library for super-fast similarity searching.
    # We will store our vectors in a FAISS "index" file.
    
    print("Step 3: Building and saving FAISS index...")
    vector_dimension = embeddings.shape[1]
    # IndexFlatL2 is a standard index for exact, nearest-neighbor search.
    index = faiss.IndexFlatL2(vector_dimension)
    # Add our vectors to the index.
    index.add(embeddings)
    # Save the index to a file for later use.
    faiss.write_index(index, 'report_index.faiss')
    print("FAISS index saved to 'report_index.faiss'")

    # --- Step 4: Create and Populate the SQLite Database ---
    # The FAISS index only stores the vectors. We need a separate database
    # to store the original text chunks that correspond to each vector.
    # The ID of each row here will match the position of the vector in FAISS.
    
    print("Step 4: Creating and populating SQLite database...")
    db_file = 'report_data.db'
    # Remove the old DB file if it exists to ensure a fresh start.
    if os.path.exists(db_file):
        os.remove(db_file)
        
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # Create a simple table with an ID and the text content.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL
        )
    ''')
    
    # Insert all our original text chunks into the database.
    # We use executemany for an efficient bulk insert.
    cursor.executemany(
        'INSERT INTO chunks (id, content) VALUES (?, ?)',
        enumerate(texts_to_embed)
    )
    
    conn.commit()
    conn.close()
    print(f"SQLite database saved to '{db_file}'")


# This is the main execution block that runs our script.
if __name__ == "__main__":
    print("Starting database build process...")
    # Step 1: Parse the report file using our function from Phase 1.
    ufdr_data = parse_ufdr('mock_ufdr.html')
    
    # Step 2: Build the vector database using the parsed data.
    build_database(ufdr_data)
    
    print("\n✅ Phase 2 Complete: Database built successfully!")
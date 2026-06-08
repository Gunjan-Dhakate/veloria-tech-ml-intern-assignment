"""
rag_search.py
=========================================================
Semantic Search using Sentence Transformers + ChromaDB

Task 3 - Veloria Tech AI/ML Internship Assignment

Features:
1. Load cricket match records from match_data.csv
2. Convert rows into natural language documents
3. Generate embeddings using all-MiniLM-L6-v2
4. Store embeddings in ChromaDB
5. Perform semantic search
6. Return Top 3 most relevant matches

Author: Gunjan Dhakate
=========================================================
"""

import os
import logging
import pandas as pd
import chromadb

from sentence_transformers import SentenceTransformer

# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = r"C:\Users\gunja\Downloads\veloria-ml-assignment_1\veloria-tech-ml-intern-assignment"

CSV_FILE = os.path.join(
    BASE_DIR,
    "match_data.csv"
)

CHROMA_DB_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)

LOG_FILE = os.path.join(
    BASE_DIR,
    "rag.log"
)

# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =====================================================
# LOAD DATA
# =====================================================

def load_match_data():
    """
    Load cricket match dataset.
    """

    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(
            f"CSV file not found: {CSV_FILE}"
        )

    df = pd.read_csv(CSV_FILE)

    print(f"\nLoaded {len(df)} match records")

    return df


# =====================================================
# CREATE DOCUMENTS
# =====================================================

def create_documents(df):
    """
    Convert match records into natural language.
    """

    documents = []

    for _, row in df.iterrows():

        document = (
            f"{row['team_1']} vs {row['team_2']} "
            f"at {row['venue']} "
            f"on {row['match_date']}. "
            f"{row['winner']} won the match. "
            f"Top scorer was {row['top_scorer']} "
            f"with {row['top_scorer_runs']} runs."
        )

        documents.append(document)

    return documents


# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

def load_embedding_model():
    """
    Load Sentence Transformer model.
    """

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Embedding model loaded.")

    return model


# =====================================================
# BUILD VECTOR DATABASE
# =====================================================

def build_vector_store(
    documents,
    model
):
    """
    Create ChromaDB vector store.
    """

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        documents,
        show_progress_bar=True
    )

    print("Embeddings generated.")

    client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR
    )

    collection_name = "ipl_matches"

    try:
        client.delete_collection(
            collection_name
        )
    except:
        pass

    collection = client.create_collection(
        name=collection_name
    )

    collection.add(
        ids=[
            str(i)
            for i in range(len(documents))
        ],
        documents=documents,
        embeddings=embeddings.tolist()
    )

    print(
        f"\nStored {len(documents)} documents in ChromaDB"
    )

    return collection


# =====================================================
# SEMANTIC SEARCH
# =====================================================

def semantic_search(
    query,
    model,
    collection,
    top_k=3
):
    """
    Search for most relevant matches.
    """

    query_embedding = model.encode(
        query
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )

    return results


# =====================================================
# DISPLAY RESULTS
# =====================================================

def display_results(results):
    """
    Print search results.
    """

    print("\n" + "=" * 60)
    print("TOP MATCH RESULTS")
    print("=" * 60)

    documents = results["documents"][0]

    for rank, document in enumerate(
        documents,
        start=1
    ):
        print(f"\nResult {rank}")
        print("-" * 40)
        print(document)


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 60)
    print("VELORIA TECH - TASK 3")
    print("Semantic Search with RAG")
    print("=" * 60)

    df = load_match_data()

    documents = create_documents(df)

    print(
        f"\nCreated {len(documents)} documents"
    )

    model = load_embedding_model()

    collection = build_vector_store(
        documents,
        model
    )

    print("\nExample Queries:")
    print("- Show matches where Chennai Super Kings won")
    print("- Show matches played at Chennai")
    print("- Find games where top scorer scored 100 runs")
    print("- Show matches against Mumbai Indians")

    while True:

        query = input(
            "\nEnter Query (or type 'exit'): "
        )

        if query.lower() == "exit":
            print("\nExiting Semantic Search...")
            break

        try:

            results = semantic_search(
                query=query,
                model=model,
                collection=collection,
                top_k=3
            )

            display_results(results)

            logging.info(
                f"Query: {query}"
            )

        except Exception as e:

            logging.error(
                f"Search Error: {e}"
            )

            print(
                f"\nSearch failed: {e}"
            )


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
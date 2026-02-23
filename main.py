import os
import requests
import time
import numpy as np
import pandas as pd
from numpy.linalg import norm

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import (
    TokenTextSplitter,
    SemanticSplitterNodeParser,
    SentenceWindowNodeParser,
)
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# CONFIG

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/tinyshakespeare.txt"

QUERIES = [
    "Who are the two feuding houses?",
    "Who is Romeo in love with?",
    "Which play contains the line 'To be, or not to be'?"
]


# DATASET

def download_dataset():
    if not os.path.exists(DATA_PATH):
        print("Downloading Tiny Shakespeare dataset...")
        os.makedirs("data", exist_ok=True)
        response = requests.get(DATA_URL)
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            f.write(response.text)
        print("Download complete.")
    else:
        print("Dataset already exists.")


def load_dataset():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()

# CHUNKING METHODS

def token_chunking(text):
    print("\n===== TOKEN CHUNKING =====")

    splitter = TokenTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )

    nodes = splitter.get_nodes_from_documents([Document(text=text)])

    avg_len = np.mean([len(n.text) for n in nodes])

    print(f"Number of chunks: {len(nodes)}")
    print(f"Average chunk length: {avg_len:.2f}")

    return nodes, avg_len


def semantic_chunking(text, embed_model):
    print("\n===== SEMANTIC CHUNKING =====")

    splitter = SemanticSplitterNodeParser(
        buffer_size=3,
        embed_model=embed_model
    )

    nodes = splitter.get_nodes_from_documents([Document(text=text)])

    avg_len = np.mean([len(n.text) for n in nodes])

    print(f"Number of chunks: {len(nodes)}")
    print(f"Average chunk length: {avg_len:.2f}")

    return nodes, avg_len


def sentence_window_chunking(text):
    print("\n===== SENTENCE WINDOW CHUNKING =====")

    splitter = SentenceWindowNodeParser.from_defaults(
        window_size=3
    )

    nodes = splitter.get_nodes_from_documents([Document(text=text)])

    avg_len = np.mean([len(n.text) for n in nodes])

    print(f"Number of chunks: {len(nodes)}")
    print(f"Average chunk length: {avg_len:.2f}")

    return nodes, avg_len


# RETRIEVAL PIPELINE

def retrieval_pipeline(nodes, embed_model, technique_name):

    print(f"\n==============================")
    print(f"   PIPELINE: {technique_name}")
    print(f"==============================")

    vector_store = SimpleVectorStore()
    Settings.embed_model = embed_model
    index = VectorStoreIndex(nodes, vector_store=vector_store)
    retriever = index.as_retriever(similarity_top_k=5)

    metrics_list = []

    for query in QUERIES:
        print(f"\n----- Query: {query} -----")

        query_embedding = np.array(embed_model.get_text_embedding(query))

        print(f"Embedding dimension: {len(query_embedding)}")
        print(f"First 8 values: {query_embedding[:8]}")
        print(f"Query vector shape: {query_embedding.shape}")

        start = time.time()
        retrieved_nodes = retriever.retrieve(query)
        latency = (time.time() - start) * 1000

        results = []
        doc_embeddings = []

        for rank, node in enumerate(retrieved_nodes, start=1):
            chunk_text = node.node.text
            store_score = node.score

            chunk_embedding = np.array(embed_model.get_text_embedding(chunk_text))
            doc_embeddings.append(chunk_embedding)

            cosine_sim = np.dot(query_embedding, chunk_embedding) / (
                norm(query_embedding) * norm(chunk_embedding)
            )

            results.append({
                "rank": rank,
                "store_score": store_score,
                "cosine_sim": cosine_sim,
                "chunk_len": len(chunk_text),
                "preview": chunk_text[:120].replace("\n", " ")
            })

        doc_embeddings = np.vstack(doc_embeddings)

        print(f"Retrieval latency: {latency:.2f} ms")
        print(f"Stacked document vectors shape: {doc_embeddings.shape}")

        df = pd.DataFrame(results)
        print(df)

        metrics_list.append({
            "Technique": technique_name,
            "Query": query,
            "Top-1 Cosine": df.iloc[0]["cosine_sim"],
            "Mean@k Cosine": df["cosine_sim"].mean(),
            "Latency (ms)": latency
        })

    return metrics_list


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    download_dataset()
    text = load_dataset()

    print("\n===== DATASET SANITY CHECK =====")
    print(f"Total characters: {len(text)}")
    print(f"Preview: {text[:300]}")

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # TOKEN
    token_nodes, token_avg = token_chunking(text)
    token_metrics = retrieval_pipeline(token_nodes, embed_model, "Token")

    # SEMANTIC
    semantic_nodes, semantic_avg = semantic_chunking(text, embed_model)
    semantic_metrics = retrieval_pipeline(semantic_nodes, embed_model, "Semantic")

    # SENTENCE WINDOW
    sentence_nodes, sentence_avg = sentence_window_chunking(text)
    sentence_metrics = retrieval_pipeline(sentence_nodes, embed_model, "Sentence-Window")

    # FINAL COMPARISON
    print("\n===== FINAL COMPARISON TABLE =====")
    final_df = pd.DataFrame(token_metrics + semantic_metrics + sentence_metrics)
    print(final_df)
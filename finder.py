import json
import faiss
from sentence_transformers import SentenceTransformer
from collections import defaultdict

model = SentenceTransformer("all-MiniLM-L6-v2")

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(BASE_DIR, "faiss_index.bin")
metadata_path = os.path.join(BASE_DIR, "metadata_index.json")

index = faiss.read_index(index_path)

with open(metadata_path) as f:
    metadata_index = json.load(f)

# load metadata
with open(metadata_path) as f:
    metadata_index = json.load(f)

# fix key type
metadata_index = {int(k): v for k, v in metadata_index.items()}


def find_similar_chunks(query, top_k=20):
    query_vector = model.encode([query])
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        
        metadata = metadata_index[idx]
        
        results.append({
            "file": metadata["file"],
            "chunk_id": metadata["chunk_id"],
            "distance": dist
        })
    
    return results


def group_by_file(results):
    file_scores = defaultdict(list)
    
    for r in results:
        file_scores[r["file"]].append(r["distance"])
    
    ranked = []
    for file, distances in file_scores.items():
        ranked.append({
            "file": file,
            "score": min(distances)
        })
    
    ranked.sort(key=lambda x: x["score"])
    return ranked
import sys
if __name__ == "__main__":
    query = " ".join(sys.argv[1:])

    chunks = find_similar_chunks(query)
    files = group_by_file(chunks)

    print(f"\nQuery: {query}\n")

    for i, item in enumerate(files[:5], 1):
        print(f"{i}. {item['file']}")
        print(f"   score: {item['score']:.4f}")
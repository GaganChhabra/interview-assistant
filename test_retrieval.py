from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("question_index.faiss")

with open("question_records.pkl", "rb") as f:
    records = pickle.load(f)

query = "capital budget alignment with corporate strategy"
query_vector = model.encode([query], normalize_embeddings=True)
query_vector = np.asarray(query_vector, dtype="float32")

scores, indices = index.search(query_vector, 5)

print("Top 5 matches for:", query)
for score, idx in zip(scores[0], indices[0]):
    print(f"[{score:.3f}]", records[idx]["question"])
    
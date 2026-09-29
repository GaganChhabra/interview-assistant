from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

load_dotenv()
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("question_index.faiss")

with open("question_records.pkl", "rb") as f:
    records = pickle.load(f)
with open("resume.txt", "r") as f:
    resume_text = f.read()
with open("jd.txt", "r") as f:
    jd_text = f.read()

resume_lines = [line.strip() for line in resume_text.split("\n") if line.strip()]
jd_lines = [line.strip() for line in jd_text.split("\n") if line.strip()]
resume_vectors = model.encode(resume_lines, normalize_embeddings=True)
jd_vectors = model.encode(jd_lines, normalize_embeddings=True)

gaps = []
for jd_line, jd_vec in zip(jd_lines, jd_vectors):
    similarities = resume_vectors @ jd_vec
    if float(np.max(similarities)) < 0.5:
        gaps.append(jd_line)

relevance_threshold = 0.3
retrieval_scores = []

print("Retrieval results:")
for focus_area in gaps[:10]:
    query_vector = model.encode([focus_area], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")
    scores, indices = index.search(query_vector, 1)
    similarity = float(scores[0][0])
    retrieval_scores.append(similarity)
    retrieved_question = records[indices[0][0]]["question"]
    is_relevant = "RELEVANT" if similarity >= relevance_threshold else "weak match"
    print(f"[{similarity:.3f}] ({is_relevant}) {retrieved_question}")

relevant_count = sum(1 for s in retrieval_scores if s >= relevance_threshold)
precision = relevant_count / len(retrieval_scores)
avg_similarity = sum(retrieval_scores) / len(retrieval_scores)

print("\nRetrieval Precision:", relevant_count, "/", len(retrieval_scores), "=", round(precision * 100, 1))
print("Average retrieval similarity:", round(avg_similarity, 3))
print("RE Score (0-100):", round(precision * 100, 1))

from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from openai import OpenAI

import faiss
import pickle
import numpy as np

load_dotenv()
token = os.getenv("HF_TOKEN")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

generated_questions = []
topics_covered = set()

for focus_area in gaps[:10]:
    query_vector = model.encode([focus_area], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")
    scores, indices = index.search(query_vector, 1)
    reference_question = records[indices[0][0]]["question"]
    topics_covered.add(records[indices[0][0]]["topic"])

    prompt = "Using this real interview question as style/content inspiration: \"" + reference_question + "\", ask one intermediate-level interview question based on this job requirement: \"" + focus_area + "\". Only output the question, nothing else."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )
    question = response.choices[0].message.content
    generated_questions.append({"question": question, "focus_area": focus_area})

relevance_scores = []
for item in generated_questions:
    q_vec = model.encode([item["question"]], normalize_embeddings=True)
    f_vec = model.encode([item["focus_area"]], normalize_embeddings=True)
    similarity = float(np.dot(q_vec[0], f_vec[0]))
    relevance_scores.append(similarity)

avg_relevance = sum(relevance_scores) / len(relevance_scores)
qa_score = avg_relevance * 100

print("Generated", len(generated_questions), "questions.")
print("Average relevance similarity:", round(avg_relevance, 3))
print("Unique topics covered:", len(topics_covered), "out of", len(generated_questions))
print("\nQA Score (0-100):", round(qa_score, 1))

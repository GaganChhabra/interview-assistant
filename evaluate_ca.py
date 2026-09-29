from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import faiss
import pickle
import numpy as np
import json

load_dotenv()
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

conversation_history = []
generated_questions = []

for focus_area in gaps[:5]:
    query_vector = model.encode([focus_area], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")
    scores, indices = index.search(query_vector, 1)
    reference_question = records[indices[0][0]]["question"]

    prompt = "Using this real interview question as style/content inspiration: \"" + reference_question + "\", ask one intermediate-level interview question based on this job requirement: \"" + focus_area + "\". Only output the question, nothing else."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )
    question = response.choices[0].message.content
    generated_questions.append(question)
    conversation_history.append(question)

judge_prompt = """You are judging the coherence of an interview conversation. Below are questions asked in sequence during one interview session.

Questions in order:
""" + "\n".join([str(i+1) + ". " + q for i, q in enumerate(generated_questions)]) + """
Rate from 0-100 how coherent and logically connected this sequence feels as a single interview session (not repetitive, reasonable topic flow, doesn't feel like disconnected random questions).
Respond ONLY with valid JSON: {"coherence_score": <0-100>, "reasoning": "<one sentence>"}
"""

judge_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": judge_prompt}],
    max_tokens=150,
)

judgment = json.loads(judge_response.choices[0].message.content)
print("Generated questions:")
for i, q in enumerate(generated_questions):
    print(str(i+1) + ".", q)

print("\nCA Score (0-100):", judgment["coherence_score"])
print("Reasoning:", judgment["reasoning"])


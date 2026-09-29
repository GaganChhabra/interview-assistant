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

test_cases = [
    {"answer": "I don't know, I would probably ask my manager.", "human_score": 10},
    {"answer": "Not sure, I guess I'd figure it out as I go.", "human_score": 15},
    {"answer": "I would talk to the team about it and make a plan.", "human_score": 35},
    {"answer": "I'd set up a basic project plan with some milestones and check in regularly.", "human_score": 50},
    {"answer": "I would establish a governance structure, involve key stakeholders early, and track progress against defined KPIs.", "human_score": 70},
    {"answer": "I'd form a cross-functional steering committee, define measurable success criteria, and report progress transparently to leadership on a regular cadence.", "human_score": 80},
    {"answer": "I would create a RACI matrix, run a risk assessment workshop with stakeholders, phase the rollout with clear gates, and tie every milestone to a measurable business KPI reviewed monthly with the C-suite.", "human_score": 90},
    {"answer": "Honestly not confident here, maybe look it up later.", "human_score": 10},
    {"answer": "I'd probably just wing it and adjust based on feedback.", "human_score": 25},
    {"answer": "I would build a structured communication plan, align cross-regional teams around shared OKRs, and use weekly retrospectives to catch misalignment early before it affects delivery.", "human_score": 85},
]

results = []

for i, case in enumerate(test_cases):
    focus_area = gaps[i % len(gaps)]
    query_vector = model.encode([focus_area], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")
    scores, indices = index.search(query_vector, 1)
    reference_question = records[indices[0][0]]["question"]

    q_prompt = "Using this real interview question as style/content inspiration: \"" + reference_question + "\", ask one intermediate-level interview question based on this job requirement: \"" + focus_area + "\". Only output the question, nothing else."
    q_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": q_prompt}],
        max_tokens=100,
    )
    question = q_response.choices[0].message.content

    eval_prompt = """You are an interview evaluator. Rate the candidate's answer 0-100 overall.

Question: """ + question + """
Answer: """ + case["answer"] + """

Respond ONLY with valid JSON: {"overall_score": <0-100>, "feedback": "<one sentence>"}
"""
    e_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": eval_prompt}],
        max_tokens=100,
    )
    evaluation = json.loads(e_response.choices[0].message.content)
    ai_score = evaluation["overall_score"]

    results.append({"human_score": case["human_score"], "ai_score": ai_score})
    print(f"Answer {i+1}: human={case['human_score']}, AI={ai_score}")

human_scores = [r["human_score"] for r in results]
ai_scores = [r["ai_score"] for r in results]

differences = [abs(h - a) for h, a in zip(human_scores, ai_scores)]
mean_abs_diff = sum(differences) / len(differences)

correlation_matrix = np.corrcoef(human_scores, ai_scores)
correlation = correlation_matrix[0][1]

aa_score = max(0, 100 - mean_abs_diff)

print("\nMean Absolute Difference:", round(mean_abs_diff, 1))
print("Correlation (Pearson r):", round(correlation, 3))
print("AA Score (0-100):", round(aa_score, 1))

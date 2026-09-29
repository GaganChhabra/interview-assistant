from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
import json

load_dotenv()
token = os.getenv("HF_TOKEN")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

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
    best_score = float(np.max(similarities))
    if best_score < 0.5:
        gaps.append(jd_line)

import faiss
import pickle

index = faiss.read_index("question_index.faiss")
with open("question_records.pkl", "rb") as f:
    records = pickle.load(f)

def retrieve_reference_question(gap_text):
    query_vector = model.encode([gap_text], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")
    scores, indices = index.search(query_vector, 1)
    return records[indices[0][0]]["question"]


print("Found", len(gaps), "gap areas to ask about.")

test_answers = [
    "I don't really know, I guess I would ask someone else to handle it.",
    "Not sure, maybe just wait and see what happens.",
    "I would set up a governance framework, align stakeholders early, and use a phased rollout tied to measurable KPIs.",
    "I'd form a cross-functional steering committee, define clear success metrics, and communicate progress transparently to the C-suite.",
]

current_difficulty = 3
consecutive_high = 0
consecutive_low = 0
difficulty_labels = {1: "beginner", 2: "basic", 3: "intermediate", 4: "advanced", 5: "expert"}

for turn, test_answer in enumerate(test_answers):
    focus_area = gaps[turn % len(gaps)]

    reference_question = retrieve_reference_question(focus_area)
    
    question_prompt = "Using this real interview question as style/content inspiration: \"" + reference_question + "\", ask one " + difficulty_labels[current_difficulty] + "-level interview question based on this job requirement: \"" + focus_area + "\". Only output the question, nothing else."


    q_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question_prompt}],
        max_tokens=100,
    )
    question = q_response.choices[0].message.content
    print("\n--- Turn", turn + 1, "(difficulty:", 
difficulty_labels[current_difficulty] + ") ---")
    print("Question:", question)
    print("Test answer:", test_answer)

    eval_prompt = """You are an interview evaluator. Rate the candidate's answer 0-100 overall.
Question: """ + question + """
Answer: """ + test_answer + """

Respond ONLY with valid JSON: {"overall_score": <0-100>, "feedback": "<one sentence>"}
"""
    e_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": eval_prompt}],
        max_tokens=100,
    )
    evaluation = json.loads(e_response.choices[0].message.content)
    score = evaluation["overall_score"]
    print("Score:", score, "| Feedback:", evaluation["feedback"])

    if score >= 75:
        consecutive_high += 1
        consecutive_low = 0
    elif score < 40:
        consecutive_low += 1
        consecutive_high = 0
    else:
        consecutive_high = 0
        consecutive_low = 0


    if consecutive_high >= 2 and current_difficulty < 5:
        current_difficulty += 1
        consecutive_high = 0
    elif consecutive_low >= 2 and current_difficulty > 1:
        current_difficulty -= 1
        consecutive_low = 0

    print("Next difficulty:", difficulty_labels[current_difficulty])







            
from dotenv import load_dotenv
import os
from typing import TypedDict
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import faiss
import pickle
import numpy as np
import json
from langgraph.graph import StateGraph, END

load_dotenv()
token = os.getenv("HF_TOKEN")
client = InferenceClient(token=token, provider="auto")
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

difficulty_labels = {1: "beginner", 2: "basic", 3: "intermediate", 4: "advanced", 5: "expert"}

print("Setup complete. Gap areas found:", len(gaps))

class InterviewState(TypedDict):
    turn: int
    current_difficulty: int
    consecutive_high: int
    consecutive_low: int
    focus_area: str
    reference_question: str
    question: str
    answer: str
    score: int
    feedback: str
    history: list

def retrieve_context(state):
    focus_area = gaps[state["turn"] % len(gaps)]
    query_vector = model.encode([focus_area], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")
    scores, indices = index.search(query_vector, 1)
    reference_question = records[indices[0][0]]["question"]
    return {"focus_area": focus_area, "reference_question": reference_question}

def interviewer_node(state):
    prompt = "Using this real interview question as style/content inspiration: \"" + state["reference_question"] + "\", ask one " + difficulty_labels[state["current_difficulty"]] + "-level interview question based on this job requirement: \"" + state["focus_area"] + "\". Only output the question, nothing else."
    response = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )
    return {"question": response.choices[0].message.content}

def evaluator_node(state):
    eval_prompt = """You are an interview evaluator. Rate the candidate's answer 0-100 overall.

Question: """ + state["question"] + """
Answer: """ + state["answer"] + """
Respond ONLY with valid JSON: {"overall_score": <0-100>, "feedback": "<one sentence>"}
    """
    response = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": eval_prompt}],
        max_tokens=100,
    )
    evaluation = json.loads(response.choices[0].message.content)
    return {"score": evaluation["overall_score"], "feedback":
evaluation["feedback"]}

def state_controller_node(state):
    score = state["score"]
    consecutive_high = state["consecutive_high"]
    consecutive_low = state["consecutive_low"]
    current_difficulty = state["current_difficulty"]

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

    history_entry = {
        "turn": state["turn"],
        "difficulty": difficulty_labels[current_difficulty],
        "question": state["question"],
        "score": score,
    }

    return {
        "current_difficulty": current_difficulty,
        "consecutive_high": consecutive_high,
        "consecutive_low": consecutive_low,
        "turn": state["turn"] + 1,
        "history": state["history"] + [history_entry],
    }

print("Nodes defined successfully.")

graph = StateGraph(InterviewState)

graph.add_node("retrieve", retrieve_context)
graph.add_node("interviewer", interviewer_node)
graph.add_node("evaluator", evaluator_node)
graph.add_node("state_controller", state_controller_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "interviewer")
graph.add_edge("interviewer", "evaluator")
graph.add_edge("evaluator", "state_controller")
graph.add_edge("state_controller", END)

app = graph.compile()

test_answers = [
    "I don't really know, I guess I would ask someone else to handle it.",
    "Not sure, maybe just wait and see what happens.",
    "I would set up a governance framework, align stakeholders early, and use a phased rollout tied to measurable KPIs.",
    "I'd form a cross-functional steering committee, define clear success metrics, and communicate progress transparently to the C-suite.",
]

state = {
    "turn": 0,
    "current_difficulty": 3,
    "consecutive_high": 0,
    "consecutive_low": 0,
    "history": [],
}

for answer in test_answers:
    state["answer"] = answer
    state = app.invoke(state)
    last = state["history"][-1]
    print("\n--- Turn", last["turn"], "(difficulty:", last["difficulty"], ") ---")
    print("Question:", last["question"])
    print("Answer:", answer)
    print("Score:", last["score"])

print("\nFinal difficulty:", difficulty_labels[state["current_difficulty"]])









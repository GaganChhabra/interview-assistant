import streamlit as st
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from openai import OpenAI

import faiss
import pickle
import numpy as np
import json

load_dotenv()
token = os.getenv("HF_TOKEN")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("question_index.faiss")
    with open("question_records.pkl", "rb") as f:
        records = pickle.load(f)
    return model, index, records

model, index, records = load_resources()

difficulty_labels = {1: "beginner", 2: "basic", 3: "intermediate", 4: "advanced", 5: "expert"}

st.title("AI Interview Preparation Assistant")

resume_text = st.text_area("Paste your resume text here", height=200)
jd_text = st.text_area("Paste the job description text here", height=200)

if "difficulty" not in st.session_state:
    st.session_state.difficulty = 3
    st.session_state.consecutive_high = 0
    st.session_state.consecutive_low = 0
    st.session_state.turn = 0

if st.button("Start Interview") and resume_text and jd_text:
    resume_lines = [line.strip() for line in resume_text.split("\n") if line.strip()]
    jd_lines = [line.strip() for line in jd_text.split("\n") if line.strip()]
    resume_vectors = model.encode(resume_lines, normalize_embeddings=True)
    jd_vectors = model.encode(jd_lines, normalize_embeddings=True)

    gaps = []
    for jd_line, jd_vec in zip(jd_lines, jd_vectors):
        similarities = resume_vectors @ jd_vec
        if float(np.max(similarities)) < 0.5:
            gaps.append(jd_line)

    st.session_state.gaps = gaps
    focus_area = gaps[0] if gaps else jd_lines[0]

    query_vector = model.encode([focus_area], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")
    scores, indices = index.search(query_vector, 1)
    reference_question = records[indices[0][0]]["question"]

    prompt = "Using this real interview question as style/content inspiration: \"" + reference_question + "\", ask one " + difficulty_labels[st.session_state.difficulty] + "-level interview question based on this job requirement: \"" + focus_area + "\". Only output the question, nothing else."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )
    st.session_state.current_question = response.choices[0].message.content
    st.session_state.current_focus_area = focus_area

if "current_question" in st.session_state:
    st.subheader("Current difficulty: " + difficulty_labels[st.session_state.difficulty])
    st.write(st.session_state.current_question)

    user_answer = st.text_area("Your answer:", key="answer_box_" + str(st.session_state.turn))
    if st.button("Submit Answer"):
        eval_prompt = """You are an interview evaluator. Rate the candidate's answer 0-100 overall.
Question: """ + st.session_state.current_question + """
Answer: """ + user_answer + """

Respond ONLY with valid JSON: {"overall_score": <0-100>, "feedback": "<one sentence>"}
"""
        eval_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": eval_prompt}],
            max_tokens=100,
        )
        evaluation = json.loads(eval_response.choices[0].message.content)
        score = evaluation["overall_score"]

        st.session_state.last_score = score
        st.session_state.last_feedback = evaluation["feedback"]
    if "last_score" in st.session_state:
        st.write("Score:", st.session_state.last_score)
        st.write("Feedback:", st.session_state.last_feedback)

        if st.button("Next Question"):
            score = st.session_state.last_score
            if score >= 75:
                st.session_state.consecutive_high += 1
                st.session_state.consecutive_low = 0

            elif score < 40:
                st.session_state.consecutive_low += 1
                st.session_state.consecutive_high = 0

            else:
                st.session_state.consecutive_high = 0
                st.session_state.consecutive_low = 0

            if st.session_state.consecutive_high >= 2 and st.session_state.difficulty < 5:
                st.session_state.difficulty += 1
                st.session_state.consecutive_high = 0
            elif st.session_state.consecutive_low >= 2 and st.session_state.difficulty > 1:
                st.session_state.difficulty -= 1
                st.session_state.consecutive_low = 0

            st.session_state.turn += 1
            gaps = st.session_state.gaps
            next_focus = gaps[st.session_state.turn % len(gaps)] if gaps else st.session_state.current_focus_area

            query_vector = model.encode([next_focus], normalize_embeddings=True)
            query_vector = np.asarray(query_vector, dtype="float32")
            scores, indices = index.search(query_vector, 1)
            next_reference = records[indices[0][0]]["question"]

            next_prompt = "Using this real interview question as style/content inspiration: \"" + next_reference + "\", ask one " + difficulty_labels[st.session_state.difficulty] + "-level interview question based on this job requirement: \"" + next_focus + "\". Only output the question, nothing else."
            next_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": next_prompt}],
                max_tokens=100,
            )
            st.session_state.current_question = next_response.choices[0].message.content
            st.session_state.current_focus_area = next_focus
            del st.session_state["last_score"]
            st.rerun()
            
    

    

        
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

with open("resume.txt", "r") as f:
    resume_text = f.read()

with open("jd.txt", "r") as f:
    jd_text = f.read()

print("Resume loaded, length:", len(resume_text))
print("JD loaded, length:", len(jd_text))
model = SentenceTransformer("all-MiniLM-L6-v2")

resume_lines = [line.strip() for line in resume_text.split("\n") if line.strip()]
jd_lines = [line.strip() for line in jd_text.split("\n") if line.strip()]

resume_vectors = model.encode(resume_lines, normalize_embeddings=True)
jd_vectors = model.encode(jd_lines, normalize_embeddings=True)

matched = []
gaps = []
threshold = 0.5

for jd_line, jd_vec in zip(jd_lines, jd_vectors):
    similarities = resume_vectors @ jd_vec
    best_score = float(np.max(similarities))
    if best_score >= threshold:
        matched.append(jd_line)
    else:
        gaps.append(jd_line)


print("\nMatched JD lines (found in resume):")
for m in matched[:5]:
    print(" -", m)


print("\nGap JD lines (not found in resume):")
for g in gaps[:5]:
    print(" -", g)

from huggingface_hub import InferenceClient
token = os.getenv("HF_TOKEN")
client = InferenceClient(token=token, provider="auto")

focus_area = gaps[0] if gaps else matched[0]
prompt = "Based on this job requirement: \"" + focus_area + "\", ask me one interview question that tests whether I can meet this requirement."
response = client.chat_completion(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=150,
)
    
print("\nPersonalized interview question:")
print(response.choices[0].message.content)





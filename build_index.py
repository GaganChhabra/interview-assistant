from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import re
import numpy as np
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

ds1 = load_dataset("ali-alkhars/interviews")
ds2 = load_dataset("K-areem/AI-Interview-Questions")

clean_questions = []

for row in ds1["train"]:
    clean_questions.append({
        "question": row["response"],
        "topic": row["input"],
        "source": "ali-alkhars/interviews",
    })

for row in ds2["train"]:
    match = re.search(r"\[INST\](.*?)\[/INST\]", row["text"], re.DOTALL)
    if match:
        question_text = match.group(1).strip()
        clean_questions.append({
            "question": question_text,
            "topic": "General",
            "source": "K-areem/AI-Interview-Questions",
        })

print("Embedding", len(clean_questions), "questions... this will take a minute or two.")

texts = [q["question"] for q in clean_questions]
vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
vectors = np.asarray(vectors, dtype="float32")

dim = vectors.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(vectors)

faiss.write_index(index, "question_index.faiss")
with open("question_records.pkl", "wb") as f:
    pickle.dump(clean_questions, f)

print("Done! Saved question_index.faiss and question_records.pkl")

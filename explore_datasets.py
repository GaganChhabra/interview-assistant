from datasets import load_dataset
import re

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

print("Total cleaned questions:", len(clean_questions))
print("Sample entries:")
for q in clean_questions[:3]:
    print(q)
for q in clean_questions[-3:]:
    print(q)
    
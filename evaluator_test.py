from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient
import json

load_dotenv()
token = os.getenv("HF_TOKEN")
client = InferenceClient(token=token, provider="auto")

question = "How would you align this initiative with the company's long-term corporate strategies?"
answer = "I would talk to stakeholders and make a plan that fits the company goals."
rubric_prompt = """You are an interview evaluator. Rate the candidate's answer on a scale of 0-100 for each criterion, and give short feedback.
Question: """ + question + """
Candidate's Answer: """ + answer + """
Respond ONLY with valid JSON in this exact format, nothing else:
{"technical_accuracy": <0-100>, "clarity": <0-100>, "depth": <0-100>,
"overall_score": <0-100>, "feedback": "<one sentence>"}
"""

response = client.chat_completion(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": rubric_prompt}],
    max_tokens=200,
)

result_text = response.choices[0].message.content
print("Raw model output:")
print(result_text)

evaluation = json.loads(result_text)
print("\nParsed evaluation:")
print(evaluation)

score = evaluation["overall_score"]
current_difficulty = 3
consecutive_high = 0
consecutive_low = 0

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
elif consecutive_low >= 2 and current_difficulty > 1:
    current_difficulty -= 1

print("\nScore:", score)
print("Difficulty stays at:", current_difficulty, "(need 2 consecutive low/high scores to shift)")



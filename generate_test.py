from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient

load_dotenv()
token = os.getenv("HF_TOKEN")
client = InferenceClient(token=token, provider="auto")

response = client.chat_completion(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Ask me one beginner-level interview question about React."}],
    max_tokens=100,
)

print(response.choices[0].message.content)



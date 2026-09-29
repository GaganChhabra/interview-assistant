from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
sentence = "I have three years of experience with React."
embedding = model.encode(sentence)
print("Sentence:", sentence)
print("Embedding length:", len(embedding))
print("First 5 numbers:", embedding[:5])

from dotenv import load_dotenv
import os
import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("question_index.faiss")

with open("question_records.pkl", "rb") as f:
    records = pickle.load(f)
with open("resume.txt", "r") as f:
    resume_1 = f.read()
with open("jd.txt", "r") as f:
    jd_1 = f.read()

resume_2 = """Junior Software Engineer with 1.5 years of experience building web applications using JavaScript and React.
Built and maintained REST API endpoints in Node.js and Express for an internal HR tool.
Wrote unit tests with Jest and participated in weekly code reviews.
Used Git and GitHub daily for version control and pull request workflows.
Worked with PostgreSQL for basic queries and schema updates.
BSc in Computer Science, graduated 2024.
Completed a capstone project building a student marks portal with a small team.
Comfortable with Agile sprints and daily standups."""

jd_2 = """We are looking for a Backend Developer to design and scale high-throughput microservices serving millions of requests per day.
Design and implement distributed systems using event-driven architecture with Kafka or RabbitMQ.
Own the deployment pipeline using Docker, Kubernetes, and CI/CD tooling on AWS.
Optimize database performance, including indexing strategies, query tuning, and sharding for large datasets.
Implement caching layers with Redis and design for horizontal scalability.
Lead incident response, write postmortems, and improve observability with Prometheus and Grafana.
Mentor junior engineers and drive technical design reviews across teams.
Define API standards, authentication with OAuth2, and security best practices.
Collaborate with product managers to translate business needs into scalable technical solutions.
Minimum 4 years of experience building production backend systems in Java, Go, or Python."""

resume_3 = """Data Analyst with 3 years of experience in retail analytics, reporting on sales and customer trends.
Built dashboards in Tableau and Power BI used by regional managers.
Proficient in SQL for extracting and aggregating data from large relational databases.
Used Python (pandas, matplotlib) for data cleaning and exploratory analysis.
Ran A/B tests on promotional campaigns and summarized results for stakeholders.
Automated weekly reporting with Excel macros and basic Python scripts.
MSc in Statistics.
Presented findings to non-technical audiences."""

jd_3 = """Develop and deploy machine learning models for customer churn prediction and demand forecasting.
Design end-to-end ML pipelines including feature engineering, training, validation, and monitoring in production.
Apply deep learning techniques such as neural networks and transformers to unstructured text and image data.
Work with big data platforms such as Spark and Databricks to process terabyte-scale datasets.
Use MLOps tools like MLflow and Airflow to version, deploy, and retrain models.
Conduct causal inference and experimentation design beyond standard A/B testing.
Collaborate with engineers to productionize models behind low-latency APIs.
Communicate model performance, bias, and limitations to executive stakeholders.
Stay current with research and evaluate emerging techniques such as large language models.
3+ years of experience with Python, scikit-learn, and PyTorch or TensorFlow."""

resume_4 = """HR Coordinator with 4 years of experience supporting recruitment and onboarding for a 300-person company.
Screened resumes, scheduled interviews, and coordinated with hiring managers.
Maintained employee records in the HRIS and ensured data accuracy.
Organized new-hire orientation and prepared onboarding documentation.
Assisted with payroll inputs and benefits enrollment queries.
Handled routine employee questions about leave and company policies.
Helped plan team-building events and internal communications.
BA in Business Administration."""

jd_4 = """Lead the full talent acquisition strategy, including employer branding, sourcing pipelines, and workforce planning.
Design and run performance management cycles, including calibration sessions and succession planning.
Manage employee relations cases, investigations, and disciplinary processes in compliance with labor law.
Own compensation benchmarking and total rewards strategy, partnering with finance on annual budgets.
Drive diversity, equity, and inclusion initiatives with measurable targets.
Build and lead a team of HR generalists and recruiters, setting goals and coaching for growth.
Advise senior leadership on organizational design, change management, and restructuring.
Use people analytics to track attrition, engagement, and hiring effectiveness and recommend actions.
Ensure HR policies and practices comply with regional regulations and audit requirements.
7+ years of HR experience including 2+ years in a people-management role."""

profiles = {
    "Profile 1: Senior Director, Project Management (original)": (resume_1, jd_1),
    "Profile 2: Junior Software Engineer to Backend Developer": (resume_2, jd_2),
    "Profile 3: Data Analyst to Data Scientist": (resume_3, jd_3),
    "Profile 4: HR Coordinator to HR Manager": (resume_4, jd_4),
}


def find_gaps(resume_text, jd_text):
    r_lines = [l.strip() for l in resume_text.split("\n") if l.strip()]
    j_lines = [l.strip() for l in jd_text.split("\n") if l.strip()]
    r_vecs = model.encode(r_lines, normalize_embeddings=True)
    j_vecs = model.encode(j_lines, normalize_embeddings=True)
    gaps = []
    for line, vec in zip(j_lines, j_vecs):
        if float(np.max(r_vecs @ vec)) < 0.5:
            gaps.append(line)
    return gaps


def ask(prompt, max_tokens=100):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def retrieve(focus):
    qv = model.encode([focus], normalize_embeddings=True)
    qv = np.asarray(qv, dtype="float32")
    scores, indices = index.search(qv, 1)
    return records[indices[0][0]]["question"], float(scores[0][0])


def similarity(a, b):
    va = model.encode([a], normalize_embeddings=True)[0]
    vb = model.encode([b], normalize_embeddings=True)[0]
    return float(np.dot(va, vb))


def judge(questions):
    numbered = "\n".join([str(i + 1) + ". " + q for i, q in enumerate(questions)])
    prompt = 'You are judging the coherence of an interview conversation. Below are questions asked in sequence during one interview session.\n\nQuestions in order:\n' + numbered + '\n\nRate from 0-100 how coherent and logically connected this sequence feels as a single interview session (not repetitive, reasonable topic flow, does not feel like disconnected random questions).\n\nRespond ONLY with valid JSON: {"coherence_score": <0-100>, "reasoning": "<one sentence>"}'
    text = ask(prompt, 150)
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])["coherence_score"]


results = []

for name, (resume_text, jd_text) in profiles.items():
    print("\nRunning:", name)
    gaps = find_gaps(resume_text, jd_text)[:10]
    print("  gap areas used:", len(gaps))
    if len(gaps) < 3:
        print("  too few gaps, skipping this profile")
        continue

    rag_qs = []
    base_qs = []
    rag_sims = []
    base_sims = []
    ret_sims = []

    for focus in gaps:
        ref, ret_sim = retrieve(focus)
        ret_sims.append(ret_sim)
        rag_q = ask('Using this real interview question as style/content inspiration: "' + ref + '", ask one intermediate-level interview question based on this job requirement: "' + focus + '". Only output the question, nothing else.')
        base_q = ask('Ask one intermediate-level interview question based on this job requirement: "' + focus + '". Only output the question, nothing else.')
        rag_qs.append(rag_q)
        base_qs.append(base_q)
        rag_sims.append(similarity(rag_q, focus))
        base_sims.append(similarity(base_q, focus))

    results.append({
        "name": name,
        "qa_rag": float(np.mean(rag_sims)) * 100,
        "qa_base": float(np.mean(base_sims)) * 100,
        "re": 100 * sum(1 for s in ret_sims if s >= 0.3) / len(ret_sims),
        "ca_rag": judge(rag_qs[:5]),
        "ca_base": judge(base_qs[:5]),
    })
    print("  done")

print("\n" + "=" * 60)
print("SUMMARY")
for r in results:
    print(r["name"])
    print("   QA:", round(r["qa_rag"], 1), "(RAG) vs", round(r["qa_base"], 1), "(baseline)")
    print("   CA:", r["ca_rag"], "(RAG) vs", r["ca_base"], "(baseline)")
    print("   RE:", round(r["re"], 1))

print("\nAVERAGES across", len(results), "profiles:")
print("   QA RAG:", round(float(np.mean([r["qa_rag"] for r in results])), 1))
print("   QA baseline:", round(float(np.mean([r["qa_base"] for r in results])), 1))
print("   CA RAG:", round(float(np.mean([r["ca_rag"] for r in results])), 1))
print("   CA baseline:", round(float(np.mean([r["ca_base"] for r in results])), 1))
print("   RE:", round(float(np.mean([r["re"] for r in results])), 1))

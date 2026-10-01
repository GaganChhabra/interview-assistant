# A Retrieval-Augmented Multi-Agent Interview Preparation Assistant

Master's thesis project by Gagan Chhabra (MSc Machine Learning and Artificial Intelligence).

This project is an interview practice assistant. It reads a candidate's resume and a job description, finds the skill gaps, and runs a multi-turn mock interview. Questions are generated with the help of a retrieved question bank. Answers are scored, and the difficulty of the next question adapts to how the candidate is doing.

## How it works

- **Three agents** built with LangGraph: Interviewer, Evaluator and State Controller.
- **Retrieval (RAG):** Sentence-BERT embeddings with a FAISS index over a bank of 6,945 interview questions.
- **Dynamic Difficulty State Machine (DDSM):** five difficulty levels that move up or down based on the Evaluator's scores.
- **LLM:** OpenAI gpt-4o-mini.
- **Question data:** the Hugging Face datasets `ali-alkhars/interviews` and `K-areem/AI-Interview-Questions`, cleaned and combined.

## Main files

| File | Purpose |
|------|---------|
| `app.py` | Interactive Streamlit app (resume and job description input, live interview) |
| `interview_graph.py` | The pipeline as a LangGraph graph (retrieve, interviewer, evaluator, state controller) |
| `mini_interview_loop.py` | The same pipeline as a plain multi-turn script |
| `personalized_question.py` | Resume and job description gap analysis, personalized questions |
| `explore_datasets.py` | Loads and cleans the question datasets |
| `build_index.py` | Builds the FAISS index and question records |
| `test_retrieval.py` | Tests question retrieval |
| `evaluator_test.py`, `ddsm_test.py` | Evaluator scoring and difficulty logic, tested on their own |
| `evaluate_qa.py`, `evaluate_qa_baseline.py` | Question quality (QA), RAG versus baseline |
| `evaluate_ca.py`, `evaluate_ca_baseline.py` | Conversational flow (CA), judged by an LLM |
| `evaluate_re.py` | Retrieval precision (RE) |
| `evaluate_aa.py` | Evaluator agreement with human scores (AA) |
| `evaluate_multi.py` | QA, CA and RE repeated across four candidate profiles |

## Setup

1. Create and activate a virtual environment:
```
   python3 -m venv venv
   source venv/bin/activate
```
2. Install the packages:
```
   pip install -r requirements.txt
```
3. Create a file named `.env` in the project folder with your own keys (use the variable names the scripts expect; never share this file):
```
   OPENAI_API_KEY=your-key-here
   HF_TOKEN=your-token-here
```
4. Create your own `resume.txt` and `jd.txt` in the project folder. The author's files are not included for privacy.
5. Build the question index (this creates `question_index.faiss` and `question_records.pkl`):
```
   python build_index.py
```
6. Run the app:
```
   streamlit run app.py
```
   Then open http://localhost:8501 in a browser.

## Evaluation summary

The system is evaluated with four metrics: QA (question quality), CA (conversational flow), RE (retrieval precision) and AA (Evaluator agreement with human scores). Results across four candidate profiles (one real, three synthetic):

| Metric | RAG | Baseline |
|--------|-----|----------|
| QA (average) | 75.5 | 70.4 |
| CA (average) | 78.8 | 82.5 |
| RE | 100 on all four profiles | not applicable |

AA was tested on the original profile only: 94.5, with a correlation of r = 0.975 against self-rated human scores.

## Limitations

- RAG improved question quality (about 7% relative) but not conversational flow, because both conditions use the same topic-ordering logic.
- The RE metric looks too lenient at its current threshold.
- Three of the four profiles are synthetic, and AA comes from a single person.
- Many questions in the dataset carry only a generic topic label.

## Files not included

`.env`, `resume.txt`, `jd.txt` and the FAISS index files are left out on purpose (secrets, personal data and rebuildable files).

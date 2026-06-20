# Resume Screening System

An AI-powered resume screening tool that analyzes resumes against a job description, ranks candidates by suitability, and generates professional summaries.

Built as part of a task assignment for Eterosoft Technologies.

## 🚀 Live Demo
👉 [Try the app here](https://candidate-ranking-ai.streamlit.app/)

---

## What It Does

- Accepts multiple PDF resumes and a job description as input
- Extracts skills from resumes automatically using Groq API (Llama 3.3 70B)
- Calculates a match score using TF-IDF cosine similarity and skill overlap
- Identifies matched and missing skills for each candidate
- Generates a brief AI-written summary per candidate
- Ranks all candidates from most to least suitable
- Allows downloading results as a CSV file

---

## How the Scoring Works

The match score combines two signals:

| Signal | Weight | Method |
|--------|--------|--------|
| Skill overlap | 60% | Matched skills / Total job skills |
| Text similarity | 40% | TF-IDF cosine similarity |

Candidates are labelled as:
- **Recommended** — score 70% and above
- **Consider** — score 40% to 69%
- **Not Suitable** — score below 40%

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web app | Streamlit |
| PDF parsing | pdfplumber |
| Skill extraction | Groq API (Llama 3.3 70B) |
| Match scoring | Scikit-learn TF-IDF + cosine similarity |
| Data handling | Pandas |
| Language | Python 3.x |

---

## Setup and Installation

```bash
git clone https://github.com/manishbhatt1/resume-screener.git
cd resume-screener
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the root folder:
GROQ_API_KEY=your_groq_api_key_here

Run the app:
```bash
python -m streamlit run app.py
```

---

## Usage

1. Paste the job description in the left panel
2. Upload one or more PDF resumes
3. Click **Analyze Candidates**
4. View ranked results, matched and missing skills, and AI summaries
5. Download results as CSV

---

## Project Structure
resume-screener/

app.py              - Main Streamlit application

requirements.txt    - Python dependencies

README.md           - Project documentation

.gitignore          - Excludes venv and .env from version control

---

## Notes

- Skill extraction depends on resume text being readable. Scanned image PDFs may not extract correctly.
- The Groq API free tier is used. For high volume screening, rate limits may apply.
- Match scores are relative indicators, not absolute judgments.

---

## Author

Manish Bhattarai  
BIT Undergraduate, Itahari International College  
manishbhattarai2024@gmail.com

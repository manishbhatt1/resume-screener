# app.py — AI Resume Screening System
# Eterosoft Technologies Task — Manish Bhattarai

import streamlit as st
import pandas as pd
import pdfplumber
import requests
import json
import io
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
load_dotenv()

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Screening System",
    page_icon=None,
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .score-high { color: #198754; font-weight: 700; font-size: 24px; }
    .score-mid  { color: #fd7e14; font-weight: 700; font-size: 24px; }
    .score-low  { color: #dc3545; font-weight: 700; font-size: 24px; }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    .badge-green { background: #d1e7dd; color: #0a3622; }
    .badge-red   { background: #f8d7da; color: #58151c; }
    .section-title {
        font-size: 13px;
        font-weight: 600;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Helper Functions ───────────────────────────────────────────
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def call_groq(prompt, max_tokens=400):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "llama-3.3-70b-versatile",
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=30
        )
        result = response.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"].strip()
        return ""
    except Exception as e:
        return ""

def extract_skills(text, context="resume"):
    prompt = f"""Extract all technical and professional skills from this {context}.
Return ONLY a JSON array of skill strings. No explanation, no markdown.
Example: ["Python", "Machine Learning", "SQL", "Communication"]

Text:
{text[:2000]}

JSON array:"""
    response = call_groq(prompt, max_tokens=300)
    try:
        response = response.replace("```json", "").replace("```", "").strip()
        skills = json.loads(response)
        return [s.strip() for s in skills if isinstance(s, str)]
    except:
        return []

def generate_summary(candidate_name, resume_text, job_desc, score, matched, missing):
    prompt = f"""Write a brief 2-3 sentence professional candidate summary for a recruiter.
Candidate name: {candidate_name}
Match score: {score:.0f}%
Matched skills: {", ".join(matched[:5]) if matched else "None"}
Missing skills: {", ".join(missing[:5]) if missing else "None"}

Be direct and professional. No bullet points. No markdown.
Summary:"""
    return call_groq(prompt, max_tokens=150)

def calculate_match(resume_text, job_text, resume_skills, job_skills):
    # TF-IDF cosine similarity on full text
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text[:3000], job_text[:3000]])
        text_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    except:
        text_score = 0

    # Skill overlap score
    resume_skills_lower = [s.lower() for s in resume_skills]
    job_skills_lower = [s.lower() for s in job_skills]

    matched = [s for s in job_skills if s.lower() in resume_skills_lower]
    missing = [s for s in job_skills if s.lower() not in resume_skills_lower]

    if job_skills_lower:
        skill_score = (len(matched) / len(job_skills_lower)) * 100
    else:
        skill_score = 0

    # Combined score — 60% skill match, 40% text similarity
    final_score = (skill_score * 0.6) + (text_score * 0.4)
    return round(final_score, 1), matched, missing

def score_color_class(score):
    if score >= 70:
        return "score-high"
    elif score >= 40:
        return "score-mid"
    return "score-low"

# ── App Header ─────────────────────────────────────────────────
st.markdown("<h1 style='font-size:28px; font-weight:700; margin-bottom:4px'>Resume Screening System</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#6c757d; margin-bottom:24px'>Upload resumes and a job description to rank and evaluate candidates automatically.</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Input Section ──────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("<div class='section-title'>Job Description</div>", unsafe_allow_html=True)
    job_desc_input = st.text_area(
        "Job Description",
        height=200,
        placeholder="Paste the full job description here...",
        label_visibility="collapsed"
    )

with col_right:
    st.markdown("<div class='section-title'>Resume Upload</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload Resumes",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    st.caption("Upload one or more PDF resumes. Each file name will be used as the candidate name.")

st.markdown("---")

# ── Run Analysis ───────────────────────────────────────────────
run_button = st.button("Analyze Candidates", type="primary", use_container_width=True)

if run_button:
    if not job_desc_input.strip():
        st.error("Please enter a job description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume PDF.")
    else:
        results = []

        with st.spinner("Extracting job requirements..."):
            job_skills = extract_skills(job_desc_input, context="job description")

        st.markdown(f"<div class='section-title'>Job Requirements Detected</div>", unsafe_allow_html=True)
        if job_skills:
            badges = " ".join([f"<span class='badge badge-green'>{s}</span>" for s in job_skills])
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.warning("Could not extract skills from job description. Check your API key.")

        st.markdown("---")
        st.markdown("<div class='section-title'>Processing Candidates</div>", unsafe_allow_html=True)

        progress = st.progress(0)

        for idx, file in enumerate(uploaded_files):
            candidate_name = file.name.replace(".pdf", "").replace("_", " ").replace("-", " ").title()

            with st.spinner(f"Analyzing {candidate_name}..."):
                # Extract resume text
                resume_text = extract_text_from_pdf(file)

                if not resume_text:
                    st.warning(f"{candidate_name}: Could not extract text from PDF.")
                    continue

                # Extract resume skills
                resume_skills = extract_skills(resume_text, context="resume")

                # Calculate match score
                score, matched, missing = calculate_match(
                    resume_text, job_desc_input, resume_skills, job_skills
                )

                # Generate AI summary
                summary = generate_summary(
                    candidate_name, resume_text, job_desc_input,
                    score, matched, missing
                )

                results.append({
                    "name": candidate_name,
                    "score": score,
                    "matched": matched,
                    "missing": missing,
                    "resume_skills": resume_skills,
                    "summary": summary,
                    "resume_text": resume_text
                })

            progress.progress((idx + 1) / len(uploaded_files))

        progress.empty()

        if not results:
            st.error("No results. Check that your PDFs contain readable text.")
        else:
            # Sort by score descending
            results.sort(key=lambda x: x["score"], reverse=True)

            st.markdown("---")
            st.markdown(f"<h2 style='font-size:20px; font-weight:700'>Results — {len(results)} Candidate(s) Ranked</h2>", unsafe_allow_html=True)

            # ── Ranking Table ──────────────────────────────────
            table_data = []
            for rank, r in enumerate(results, 1):
                status = "Recommended" if r["score"] >= 70 else "Consider" if r["score"] >= 40 else "Not Suitable"
                table_data.append({
                    "Rank": rank,
                    "Candidate": r["name"],
                    "Match Score (%)": r["score"],
                    "Matched Skills": len(r["matched"]),
                    "Missing Skills": len(r["missing"]),
                    "Status": status
                })

            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("<h2 style='font-size:20px; font-weight:700'>Candidate Details</h2>", unsafe_allow_html=True)

            # ── Individual Cards ───────────────────────────────
            for rank, r in enumerate(results, 1):
                with st.expander(f"#{rank} — {r['name']} — {r['score']}% Match", expanded=(rank == 1)):

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        css_class = score_color_class(r["score"])
                        st.markdown(f"<div class='section-title'>Match Score</div>", unsafe_allow_html=True)
                        st.markdown(f"<span class='{css_class}'>{r['score']}%</span>", unsafe_allow_html=True)

                    with c2:
                        st.markdown(f"<div class='section-title'>Matched Skills ({len(r['matched'])})</div>", unsafe_allow_html=True)
                        if r["matched"]:
                            badges = " ".join([f"<span class='badge badge-green'>{s}</span>" for s in r["matched"]])
                            st.markdown(badges, unsafe_allow_html=True)
                        else:
                            st.caption("None matched")

                    with c3:
                        st.markdown(f"<div class='section-title'>Missing Skills ({len(r['missing'])})</div>", unsafe_allow_html=True)
                        if r["missing"]:
                            badges = " ".join([f"<span class='badge badge-red'>{s}</span>" for s in r["missing"]])
                            st.markdown(badges, unsafe_allow_html=True)
                        else:
                            st.caption("No missing skills")

                    st.markdown("<div class='section-title' style='margin-top:16px'>AI Summary</div>", unsafe_allow_html=True)
                    if r["summary"]:
                        st.write(r["summary"])
                    else:
                        st.caption("Summary not available.")

                    st.markdown("<div class='section-title' style='margin-top:16px'>All Extracted Skills from Resume</div>", unsafe_allow_html=True)
                    if r["resume_skills"]:
                        badges = " ".join([f"<span class='badge' style='background:#e9ecef;color:#495057'>{s}</span>" for s in r["resume_skills"]])
                        st.markdown(badges, unsafe_allow_html=True)
                    else:
                        st.caption("No skills extracted.")

            # ── Download Results ───────────────────────────────
            st.markdown("---")
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="screening_results.csv",
                mime="text/csv"
            )
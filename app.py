import os
import json
import csv
import io
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, Response
from groq import Groq
from utils import extract_text, get_candidate_name, extract_candidate_name


load_dotenv()

app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Max file size — 10MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ---- ROUTES ----

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/screen", methods=["POST"])
def screen():

    # ---- STEP 1: GET THE JD TEXT ----
    jd_text = ""

    jd_file = request.files.get("jd_file")
    if jd_file and jd_file.filename != "":
        jd_text = extract_text(jd_file, jd_file.filename)

    if not jd_text:
        jd_text = request.form.get("jd_text", "").strip()

    if not jd_text:
        return jsonify({"error": "Please provide a Job Description."}), 400


    # ---- STEP 2: GET ALL RESUME FILES ----
    resume_files = request.files.getlist("resumes")

    if not resume_files or resume_files[0].filename == "":
        return jsonify({"error": "Please upload at least one resume."}), 400


    # ---- STEP 3: EXTRACT TEXT FROM EACH RESUME ----
    candidates = []

    for resume_file in resume_files:
        text = extract_text(resume_file, resume_file.filename)

        # Try AI name extraction first, fall back to filename
        ai_name = extract_candidate_name(text, client)
        name = ai_name if ai_name else get_candidate_name(text, resume_file.filename)

        candidates.append({
            "name": name,
            "filename": resume_file.filename,
            "resume_text": text
        })


    # ---- STEP 4: SCREEN EACH CANDIDATE ----
    results = []

    for candidate in candidates:
        result = screen_candidate(
            candidate["name"],
            candidate["resume_text"],
            jd_text
        )
        result["name"] = candidate["name"]
        result["filename"] = candidate["filename"]
        results.append(result)


    # ---- STEP 5: SORT BY MATCH SCORE ----
    results.sort(key=lambda x: x.get("overall_match", 0), reverse=True)

    # ---- STEP 6: EXTRACT JOB TITLE ----
    job_title = extract_job_title(jd_text)

    return render_template("results.html", results=results, job_title=job_title)


# ---- AI FUNCTIONS ----

def extract_job_title(jd_text):
    """Extract just the job title from the JD."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=20,
            temperature=0,
            messages=[{
                "role": "user",
                "content": f"""Extract only the job title from this job description.
Return just the job title, nothing else. No punctuation, no explanation.

JD:
{jd_text[:500]}"""
            }]
        )
        return response.choices[0].message.content.strip()
    except:
        return "Role"


def screen_candidate(name, resume_text, jd_text):
    """Send one resume + JD to AI. Returns structured dict."""

    prompt = f"""You are an expert HR recruiter and talent assessor with deep 
experience in resume screening and candidate evaluation.

Analyse the resume against the job description and return a structured assessment.

IMPORTANT CLASSIFICATION RULES — assign exactly one fit_category:
- "exact_fit": Candidate meets virtually all requirements directly and explicitly
- "semantic_fit": Different job titles or words but clearly same capability 
  (e.g. "People Analytics" vs "HR Data Analysis")
- "transferable_fit": Different domain but skills genuinely transfer 
  (e.g. Operations Manager applying for HR Ops role)
- "keyword_spam": Resume mirrors JD language suspiciously closely but 
  lacks genuine depth, detail, or evidence of actual experience
- "irrelevant": No meaningful overlap with the JD requirements

Pay special attention to mandatory qualifications, certifications, or 
requirements in the JD. Explicitly call out any that are missing.

For explainability, provide specific reasoning for each score and cite 
actual evidence from the resume — specific job titles, years, skills, 
certifications, or phrases that drove your assessment.

Return ONLY valid JSON. No explanation, no markdown, no backticks.
Exactly this structure:

{{
  "overall_match": <integer 0-100>,
  "skills_match": <integer 0-100>,
  "experience_match": <integer 0-100>,
  "education_match": <integer 0-100>,
  "fit_category": <"exact_fit"|"semantic_fit"|"transferable_fit"|"keyword_spam"|"irrelevant">,
  "fit_reasoning": "<one sentence explaining why this fit category was assigned>",
  "score_reasoning": {{
    "skills": "<one sentence: why this skills score, citing specific skills found or missing>",
    "experience": "<one sentence: why this experience score, citing years/roles found vs required>",
    "education": "<one sentence: why this education score, citing qualification found vs required>"
  }},
  "evidence": {{
    "supporting": ["<direct quote or specific detail from resume that supports the match>", "<another>", "<another>"],
    "against": ["<specific detail from resume or gap that works against the match>", "<another>"]
  }},
  "recommendation": "<recruiter-style 2-3 sentence narrative — what to do with this candidate and why>",
  "strengths": ["<string>", "<string>", "<string>"],
  "gaps": ["<string>", "<string>", "<string>"],
  "suggestions": ["<string>", "<string>", "<string>"],
  "mandatory_requirements_met": <true or false>,
  "mandatory_gaps": ["<string>"],
  "summary": "<2 sentence overall assessment>"
}}

JOB DESCRIPTION:
{jd_text}

RESUME — {name}:
{resume_text}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1500,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        return data

    except json.JSONDecodeError:
        return {
            "overall_match": 0,
            "skills_match": 0,
            "experience_match": 0,
            "education_match": 0,
            "fit_category": "irrelevant",
            "fit_reasoning": "Could not parse response",
            "score_reasoning": {
                "skills": "Could not parse",
                "experience": "Could not parse",
                "education": "Could not parse"
            },
            "evidence": {"supporting": [], "against": []},
            "recommendation": "Analysis failed. Please try again.",
            "strengths": ["Could not parse response"],
            "gaps": ["Could not parse response"],
            "suggestions": ["Please try again"],
            "mandatory_requirements_met": False,
            "mandatory_gaps": ["Could not parse response"],
            "summary": "Analysis failed. Please try again."
        }

    except Exception as e:
        return {
            "overall_match": 0,
            "skills_match": 0,
            "experience_match": 0,
            "education_match": 0,
            "fit_category": "irrelevant",
            "fit_reasoning": "Error during analysis",
            "score_reasoning": {
                "skills": "Error",
                "experience": "Error",
                "education": "Error"
            },
            "evidence": {"supporting": [], "against": []},
            "recommendation": "Analysis failed. Please try again.",
            "strengths": [],
            "gaps": [],
            "suggestions": [],
            "mandatory_requirements_met": False,
            "mandatory_gaps": [],
            "summary": f"Error: {str(e)}"
        }


# ---- DOWNLOAD ROUTES ----

@app.route("/download-csv", methods=["POST"])
def download_csv():
    """Receives results as JSON, returns downloadable CSV."""
    data = request.get_json()
    results = data.get("results", [])
    job_title = data.get("job_title", "Role")
    job_title = job_title.replace("\n", "").replace("\r", "")

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Candidate", "File", "Fit Category", "Overall Match %",
        "Skills Match %", "Experience Match %", "Education Match %",
        "Mandatory Requirements Met", "Mandatory Gaps",
        "Strengths", "Gaps", "Recommendation", "Summary"
    ])

    for r in results:
        writer.writerow([
            r.get("name", ""),
            r.get("filename", ""),
            r.get("fit_category", "").replace("_", " ").title(),
            r.get("overall_match", 0),
            r.get("skills_match", 0),
            r.get("experience_match", 0),
            r.get("education_match", 0),
            "Yes" if r.get("mandatory_requirements_met", True) else "No",
            " | ".join(r.get("mandatory_gaps", [])),
            " | ".join(r.get("strengths", [])),
            " | ".join(r.get("gaps", [])),
            r.get("recommendation", ""),
            r.get("summary", "")
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=screening_{job_title.replace(' ', '_')}.csv"
        }
    )


if __name__ == "__main__":
    app.run(debug=True)

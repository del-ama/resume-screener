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
    """Extract just the job title from the JD — single line, max 5 words."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=20,
            temperature=0,
            messages=[{
                "role": "user",
                "content": f"""Extract only the single job title from this job description.
Return maximum 5 words. No punctuation, no explanation, no extra lines.
Only the job title itself on one line.

JD:
{jd_text[:500]}"""
            }]
        )
        title = response.choices[0].message.content.strip()
        # Strip all newlines — critical for HTTP headers
        title = title.replace("\n", " ").replace("\r", " ")
        title = title.split("\n")[0].strip()
        return title
    except:
        return "Role"


def screen_candidate(name, resume_text, jd_text):
    """Send one resume + JD to AI. Returns structured dict."""

    prompt = f"""You are an expert HR recruiter and talent assessor with deep
experience in resume screening and candidate evaluation.

Analyse the resume against the job description and return a structured assessment.

CLASSIFICATION RULES — assign exactly one fit_category:
- "exact_fit": Candidate meets virtually all requirements directly and explicitly
- "semantic_fit": Different job titles or words but clearly the same capability
  (e.g. "People Analytics" vs "HR Data Analysis")
- "transferable_fit": Different domain but skills genuinely transfer
  (e.g. Operations Manager applying for HR Ops role)
- "keyword_spam": Resume mirrors JD language suspiciously closely but
  lacks genuine depth, detail, or evidence of actual experience
- "irrelevant": No meaningful overlap with the JD requirements

CURRENT TITLE AND COMPANY RULE:
Extract the candidate's most recent job title and employer from the resume.
If not clearly stated, return NULL for that field. Do not guess or infer.

STRENGTHS RULE:
Strengths must cite specific evidence only — project names, technologies,
measurable outcomes, role titles, years of experience.
Never write generic statements like "strong technical skills" or
"good educational background". If you cannot cite specific evidence,
do not include it as a strength.

GAPS RULE:
Gaps = all shortcomings including inferred ones. Be specific about what
is missing and why it matters for this role.

MANDATORY REQUIREMENTS RULE:
mandatory_requirements_met and mandatory_gaps apply ONLY when the JD
explicitly uses words like "required", "mandatory", "must have", "essential".
Do not infer mandatory requirements. If the JD does not explicitly mark
something as mandatory, it does not appear here even if it is a significant gap.
If no mandatory requirements are explicitly stated in the JD, set
mandatory_requirements_met to true and mandatory_gaps to an empty list.

EVIDENCE AGAINST RULE:
evidence.against must contain specific things IN the resume that actively
raise concerns — unexplained employment gaps, very frequent job changes,
claims without any supporting project evidence, inconsistencies between
skills claimed and experience described.
Do NOT restate gaps or missing skills here.
If no genuine red flags exist, return an empty list.

INTERVIEW FOCUS AREAS RULE:
Write specific, actionable questions or areas the interviewer should probe,
based on the gaps and evidence found. Write them as concrete interview prompts.
Example: "Ask candidate to walk through a specific end-to-end data analysis
project they owned independently — assess depth of contribution."
Do not write generic advice like "assess technical skills."

RECOMMENDATION RULE:
State a clear hiring action on the first line — one of:
Shortlist for [specific round] / Reject / Hold — pending [specific reason].
Follow with one sentence giving the single most important reason.
Example: "Shortlist for technical screening — Python and SQL skills are
directly relevant, but Power BI gap must be assessed before proceeding."
Example: "Reject — candidate has no data analysis experience and does not
meet the mandatory tool requirements for this role."
Example: "Hold — strong profile but appears overqualified; consider for
the senior analyst opening instead."

Return ONLY valid JSON. No explanation, no markdown, no backticks.
Exactly this structure:

{{
  "overall_match": <integer 0-100>,
  "skills_match": <integer 0-100>,
  "experience_match": <integer 0-100>,
  "education_match": <integer 0-100>,
  "current_title": "<candidate's most recent job title, or NULL if not found>",
  "current_company": "<candidate's most recent employer name, or NULL if not found>",
  "fit_category": <"exact_fit"|"semantic_fit"|"transferable_fit"|"keyword_spam"|"irrelevant">,
  "fit_reasoning": "<one sentence explaining why this fit category was assigned>",
  "score_reasoning": {{
    "skills": "<one sentence: why this skills score, citing specific skills found or missing>",
    "experience": "<one sentence: why this experience score, citing years/roles found vs required>",
    "education": "<one sentence: why this education score, citing qualification found vs required>"
  }},
  "strengths": ["<specific strength with named evidence>", "<another>", "<another>"],
  "gaps": ["<specific gap and why it matters for this role>", "<another>", "<another>"],
  "evidence": {{
    "supporting": ["<specific project, achievement, or detail from resume that supports the match>", "<another>", "<another>"],
    "against": ["<specific resume detail that raises a genuine concern, or empty list if none>"]
  }},
  "mandatory_requirements_met": <true or false>,
  "mandatory_gaps": ["<explicitly required item that is missing>"],
  "interview_focus_areas": ["<specific actionable interview question or probe>", "<another>", "<another>"],
  "recommendation": "<Shortlist for X / Reject / Hold — one sentence reason>"
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
            "current_title": "Unknown",
            "current_company": "Unknown",
            "fit_category": "irrelevant",
            "fit_reasoning": "Could not parse response",
            "score_reasoning": {
                "skills": "Could not parse",
                "experience": "Could not parse",
                "education": "Could not parse"
            },
            "strengths": ["Could not parse response"],
            "gaps": ["Could not parse response"],
            "evidence": {"supporting": [], "against": []},
            "mandatory_requirements_met": False,
            "mandatory_gaps": ["Could not parse response"],
            "interview_focus_areas": [],
            "recommendation": "Analysis failed. Please try again."
        }

    except Exception as e:
        return {
            "overall_match": 0,
            "skills_match": 0,
            "experience_match": 0,
            "education_match": 0,
            "current_title": "Unknown",
            "current_company": "Unknown",
            "fit_category": "irrelevant",
            "fit_reasoning": "Error during analysis",
            "score_reasoning": {
                "skills": "Error",
                "experience": "Error",
                "education": "Error"
            },
            "strengths": [],
            "gaps": [],
            "evidence": {"supporting": [], "against": []},
            "mandatory_requirements_met": False,
            "mandatory_gaps": [],
            "interview_focus_areas": [],
            "recommendation": f"Analysis failed. Error: {str(e)}"
        }


# ---- DOWNLOAD ROUTES ----

@app.route("/download-csv", methods=["POST"])
def download_csv():
    """
    Dynamic CSV — automatically includes every field from the AI response.
    No manual column updates needed when prompt changes.
    """
    data = request.get_json()
    results = data.get("results", [])
    job_title = data.get("job_title", "Role")

    # Strip newlines — critical, newlines in HTTP headers cause 500 errors
    job_title = job_title.replace("\n", " ").replace("\r", " ").strip()

    if not results:
        return jsonify({"error": "No results to download"}), 400

    def flatten(value):
        """Convert any value type to a clean readable string for CSV."""
        if isinstance(value, list):
            return " | ".join(str(v) for v in value)
        if isinstance(value, dict):
            return " | ".join(f"{k}: {v}" for k, v in value.items())
        if isinstance(value, bool):
            return "Yes" if value else "No"
        return str(value) if value is not None else ""

    # Build headers dynamically from all keys across all results
    all_keys = []
    seen = set()
    for r in results:
        for k in r.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(all_keys)

    for r in results:
        writer.writerow([flatten(r.get(k, "")) for k in all_keys])

    output.seek(0)

    safe_title = job_title.replace(" ", "_")
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=screening_{safe_title}.csv"
        }
    )


if __name__ == "__main__":
    app.run(debug=True)

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

EXPERIENCE BAND RULE:
Assign exactly one band based on overall career stage visible in the resume:
- "Early Career": 0-3 years total experience or fresh graduate
- "Mid Level": 3-8 years with growing ownership and specialisation
- "Senior": 8-15 years with clear domain expertise and independent delivery
- "Leadership": 15+ years or clear people/org leadership regardless of years
Do not count internships or part-time roles as full experience.

CURRENT TITLE AND COMPANY RULE:
Extract the candidate's most recent job title and employer.
If not clearly stated, return NULL. Do not guess or infer.

MANDATORY REQUIREMENTS RULE:
List every requirement the JD explicitly marks as "required", "mandatory",
"must have", or "essential". For each one, state whether the resume
meets it or not with a brief reason.
If the JD has no explicitly marked mandatory requirements, return an empty list.
Do NOT infer mandatory requirements — only use what is explicitly stated.

STRENGTHS RULE:
Strengths must cite specific evidence — project names, technologies,
measurable outcomes, role titles, years of experience.
Never write generic statements like "strong technical skills".
If you cannot cite specific evidence, do not include it.

GAPS RULE:
Gaps = all shortcomings including inferred ones.
Be specific about what is missing and why it matters for this role.

RED FLAGS RULE:
Flag any of the following if present in the resume:
- More than 2 roles shorter than 12 months (excluding internships/contracts)
- Any unexplained gap longer than 6 months between roles
- Applying significantly below current seniority level (overqualification)
- Claims in skills section with zero supporting evidence in experience
If none of these are present, return an empty list. Do not fabricate red flags.

SKILLS RULE:
List only skills that are relevant to this specific JD.
For each skill, assign exactly one context:
- "hands_on": candidate has directly used this skill in a project or role
- "oversight": candidate has managed or directed others using this skill
- "exposure": candidate mentions it in passing or education with no project evidence
Do not list skills not relevant to the JD.

INDUSTRIES AND DOMAINS RULE:
List the industries and functional domains the candidate has worked in,
based on company names and role descriptions in the resume.
Examples of industries: FMCG, Banking, Healthcare, IT Services, Consulting
Examples of domains: Data Analytics, HR Operations, Product Management, Finance

EVIDENCE AGAINST RULE:
Specific things IN the resume that actively raise concerns — not gap restatements.
Unexplained employment gaps, frequent short tenures, inconsistencies between
claimed skills and described experience. If none, return empty list.

INTERVIEW FOCUS AREAS RULE:
Specific, actionable interview prompts based on gaps found.
Example: "Ask candidate to walk through an end-to-end data analysis project
they owned independently — assess depth of contribution vs team support."
Not generic advice. Minimum 3 prompts.

RECOMMENDATION RULE:
First line: clear hiring action — Shortlist for [round] / Reject / Hold.
Second line: single most important reason.

Return ONLY valid JSON. No explanation, no markdown, no backticks.
Exactly this structure:

{{
  "overall_match": <integer 0-100>,
  "skills_match": <integer 0-100>,
  "experience_match": <integer 0-100>,
  "education_match": <integer 0-100>,
  "experience_band": <"Early Career"|"Mid Level"|"Senior"|"Leadership">,
  "current_title": "<most recent job title or NULL>",
  "current_company": "<most recent employer or NULL>",
  "fit_category": <"exact_fit"|"semantic_fit"|"transferable_fit"|"keyword_spam"|"irrelevant">,
  "fit_reasoning": "<one sentence>",
  "score_reasoning": {{
    "skills": "<one sentence citing specific skills found or missing>",
    "experience": "<one sentence citing years and roles found vs required>",
    "education": "<one sentence citing qualification found vs required>"
  }},
  "mandatory_requirements": [
    {{
      "requirement": "<the mandatory requirement from the JD>",
      "met": <true or false>,
      "reason": "<one sentence — what was found or missing in the resume>"
    }}
  ],
  "strengths": ["<specific strength with named evidence>", "<another>", "<another>"],
  "gaps": ["<specific gap and why it matters>", "<another>", "<another>"],
  "evidence": {{
    "supporting": ["<specific project or achievement from resume>", "<another>", "<another>"],
    "against": ["<specific concern from resume>"]
  }},
  "skills": [
    {{
      "skill": "<skill name>",
      "context": <"hands_on"|"oversight"|"exposure">
    }}
  ],
  "industries_and_domains": ["<industry or domain>", "<another>"],
  "red_flags": ["<specific red flag with detail>"],
  "interview_focus_areas": ["<specific interview prompt>", "<another>", "<another>"],
  "recommendation": "<Shortlist for X / Reject / Hold — reason on next line>"
}}

JOB DESCRIPTION:
{jd_text}

RESUME — {name}:
{resume_text}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=2000,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        return data

    except json.JSONDecodeError:
        return _fallback("Could not parse AI response")

    except Exception as e:
        return _fallback(f"Error: {str(e)}")


def _fallback(reason):
    """Return a safe empty result when AI response fails."""
    return {
        "overall_match": 0,
        "skills_match": 0,
        "experience_match": 0,
        "education_match": 0,
        "experience_band": "Unknown",
        "current_title": "Unknown",
        "current_company": "Unknown",
        "fit_category": "irrelevant",
        "fit_reasoning": reason,
        "score_reasoning": {
            "skills": "Could not parse",
            "experience": "Could not parse",
            "education": "Could not parse"
        },
        "mandatory_requirements": [],
        "strengths": [],
        "gaps": [],
        "evidence": {"supporting": [], "against": []},
        "skills": [],
        "industries_and_domains": [],
        "red_flags": [],
        "interview_focus_areas": [],
        "recommendation": "Analysis failed. Please try again."
    }


# ---- DOWNLOAD ROUTES ----

@app.route("/download-csv", methods=["POST"])
def download_csv():
    """
    Dynamic CSV — automatically includes every field.
    No manual updates needed when prompt changes.
    """
    data = request.get_json()
    results = data.get("results", [])
    job_title = data.get("job_title", "Role")
    job_title = job_title.replace("\n", " ").replace("\r", " ").strip()

    if not results:
        return jsonify({"error": "No results to download"}), 400

    def flatten(value):
        """Convert any value type to a clean readable string for CSV."""
        if isinstance(value, list):
            # Handle list of dicts (e.g. skills, mandatory_requirements)
            parts = []
            for item in value:
                if isinstance(item, dict):
                    parts.append(" | ".join(f"{k}: {v}" for k, v in item.items()))
                else:
                    parts.append(str(item))
            return " || ".join(parts)
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

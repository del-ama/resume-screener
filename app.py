import os
import json
import csv
import io
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, Response
from groq import Groq
from utils import extract_text, get_candidate_name, extract_candidate_name
from fpdf import FPDF

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


@app.route("/download-pdf", methods=["POST"])
def download_pdf():
    """Receives results as JSON, builds PDF using fpdf2."""
    data = request.get_json()
    results = data.get("results", [])
    job_title = data.get("job_title", "Role")

    fit_labels = {
        "exact_fit": "Exact Fit",
        "semantic_fit": "Semantic Fit",
        "transferable_fit": "Transferable Fit",
        "keyword_spam": "Keyword Spam",
        "irrelevant": "Irrelevant"
    }

    # fpdf2 uses RGB tuples for colors
    fit_colors = {
        "exact_fit": (56, 161, 105),
        "semantic_fit": (49, 130, 206),
        "transferable_fit": (214, 158, 46),
        "keyword_spam": (229, 62, 62),
        "irrelevant": (160, 174, 192)
    }

    def score_color(s):
        if s >= 70: return (56, 161, 105)
        if s >= 40: return (214, 158, 46)
        return (229, 62, 62)

    def safe(text):
        """Remove characters fpdf can't handle"""
        if not text:
            return ""
        return str(text).encode("latin-1", errors="replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    # ---- REPORT HEADER ----
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(44, 82, 130)
    pdf.cell(0, 10, "Screening Report", ln=True)

    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 8, safe(job_title), ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(113, 128, 150)
    pdf.cell(0, 6, f"{len(results)} candidate{'s' if len(results) != 1 else ''} screened", ln=True)

    pdf.ln(4)
    pdf.set_draw_color(44, 82, 130)
    pdf.set_line_width(0.8)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(8)

    # ---- CANDIDATE SECTIONS ----
    for r in results:
        cat = r.get("fit_category", "irrelevant")
        score = r.get("overall_match", 0)
        color = fit_colors.get(cat, (160, 174, 192))
        label = fit_labels.get(cat, "Unknown")
        sc = score_color(score)
        score_reasoning = r.get("score_reasoning", {})

        # Candidate name bar
        pdf.set_fill_color(247, 250, 252)
        pdf.set_draw_color(*color)
        pdf.set_line_width(1.2)
        pdf.rect(20, pdf.get_y(), 170, 16, style="FD")

        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(45, 55, 72)
        pdf.set_xy(23, pdf.get_y() + 3)
        pdf.cell(100, 6, safe(r.get("name", "")))

        # Score circle (approximated as colored text)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*sc)
        pdf.set_xy(155, pdf.get_y())
        pdf.cell(30, 6, f"{score}%", align="R")
        pdf.ln(13)

        # Filename and fit category
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(160, 174, 192)
        pdf.cell(0, 5, safe(r.get("filename", "")), ln=True)

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*color)
        pdf.cell(0, 5, label, ln=True)
        pdf.ln(2)

        # Fit reasoning
        if r.get("fit_reasoning"):
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(113, 128, 150)
            pdf.multi_cell(0, 5, safe(r.get("fit_reasoning", "")))
            pdf.ln(2)

        # Score bars (text-based)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(74, 85, 104)
        for bar_label, val in [
            ("Skills Match", r.get("skills_match", 0)),
            ("Experience Match", r.get("experience_match", 0)),
            ("Education Match", r.get("education_match", 0))
        ]:
            bc = score_color(val)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(74, 85, 104)
            pdf.cell(45, 5, bar_label)

            # Draw bar track
            bar_x = pdf.get_x()
            bar_y = pdf.get_y() + 1
            pdf.set_fill_color(237, 242, 247)
            pdf.rect(bar_x, bar_y, 100, 4, style="F")

            # Draw bar fill
            pdf.set_fill_color(*bc)
            fill_width = val
            if fill_width > 0:
                pdf.rect(bar_x, bar_y, fill_width, 4, style="F")

            pdf.set_xy(bar_x + 102, pdf.get_y())
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*bc)
            pdf.cell(20, 5, f"{val}%", ln=True)

        pdf.ln(3)

        # Score Reasoning box
        pdf.set_fill_color(235, 244, 255)
        pdf.set_draw_color(235, 244, 255)
        y_before = pdf.get_y()
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(44, 82, 130)
        pdf.cell(0, 5, "Score Reasoning:", ln=True)
        for key, field in [
            ("Skills", "skills"),
            ("Experience", "experience"),
            ("Education", "education")
        ]:
            val = score_reasoning.get(field, "")
            if val:
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(74, 85, 104)
                pdf.cell(22, 5, f"{key}:")
                pdf.set_font("Helvetica", "", 9)
                pdf.multi_cell(0, 5, safe(val))
        pdf.ln(2)

        # Mandatory gaps
        if not r.get("mandatory_requirements_met", True):
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(229, 62, 62)
            pdf.cell(0, 5, "Mandatory Requirements Not Met:", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(74, 85, 104)
            for gap in r.get("mandatory_gaps", []):
                pdf.cell(5, 5, "")
                pdf.multi_cell(0, 5, safe(f"• {gap}"))
            pdf.ln(2)

        # Two column sections
        sections = [
            ("Strengths", r.get("strengths", []), (56, 161, 105)),
            ("Gaps", r.get("gaps", []), (229, 62, 62)),
            ("Suggestions", r.get("suggestions", []), (49, 130, 206)),
        ]

        for sec_label, items, sec_color in sections:
            if items:
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(*sec_color)
                pdf.cell(0, 5, sec_label + ":", ln=True)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(74, 85, 104)
                for item in items:
                    pdf.cell(5, 5, "")
                    pdf.multi_cell(0, 5, safe(f"• {item}"))
                pdf.ln(1)

        # Supporting / Against Evidence
        supporting = r.get("evidence", {}).get("supporting", [])
        against = r.get("evidence", {}).get("against", [])

        if supporting:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(56, 161, 105)
            pdf.cell(0, 5, "Supporting Evidence:", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(74, 85, 104)
            for e in supporting:
                pdf.cell(5, 5, "")
                pdf.multi_cell(0, 5, safe(f"✓ {e}"))
            pdf.ln(1)

        if against:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(229, 62, 62)
            pdf.cell(0, 5, "Evidence Against:", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(74, 85, 104)
            for e in against:
                pdf.cell(5, 5, "")
                pdf.multi_cell(0, 5, safe(f"✗ {e}"))
            pdf.ln(1)

        # Recommendation
        rec = r.get("recommendation", "")
        if rec:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(56, 161, 105)
            pdf.cell(0, 5, "Recruiter Recommendation:", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(74, 85, 104)
            pdf.multi_cell(0, 5, safe(rec))
            pdf.ln(1)

        # Summary
        summary = r.get("summary", "")
        if summary:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(44, 82, 130)
            pdf.cell(0, 5, "Summary:", ln=True)
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(74, 85, 104)
            pdf.multi_cell(0, 5, safe(summary))

        # Divider between candidates
        pdf.ln(4)
        pdf.set_draw_color(226, 232, 240)
        pdf.set_line_width(0.3)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(8)

    # Output PDF as bytes
    pdf_bytes = pdf.output()

    return Response(
        bytes(pdf_bytes),
        mimetype="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=screening_{job_title.replace(' ', '_')}.pdf"
        }
    )

if __name__ == "__main__":
    app.run(debug=True)
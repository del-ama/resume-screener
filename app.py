import os
import json
import csv
import io
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, Response, redirect, url_for, session
from groq import Groq
from utils import extract_text, get_candidate_name, extract_candidate_name

# For testing railway errors
import sys
print(f"GROQ_API_KEY present: {bool(os.getenv('GROQ_API_KEY'))}", file=sys.stderr)

load_dotenv()

app = Flask(__name__)

# Secret key is required for Flask sessions to work.
# A session is like a secure cookie — Flask uses this key to sign it
# so it can't be tampered with. Set APP_SECRET_KEY in Railway Variables.
# If not set, we fall back to a random key (sessions won't persist across restarts).
app.secret_key = os.getenv("APP_SECRET_KEY", os.urandom(24))

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Max file size — 10MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ---- USER LOADING ----

def load_users():
    """
    Reads APP_USERS from environment and returns a dict of {username: password}.

    In Railway Variables, set APP_USERS like this:
        amar:mypassword,client1:theirpass,client2:anotherpass

    Each user is separated by a comma. Username and password are separated by a colon.
    """
    users = {}
    raw = os.getenv("APP_USERS", "")
    if not raw:
        return users

    for entry in raw.split(","):
        entry = entry.strip()
        if ":" in entry:
            username, password = entry.split(":", 1)  # Split on first colon only
            users[username.strip()] = password.strip()

    return users


# ---- LOGIN CHECK HELPER ----

def is_logged_in():
    """Returns True if the current user has a valid session."""
    return session.get("logged_in") is True


# ---- LOGIN / LOGOUT ROUTES ----

@app.route("/login", methods=["GET", "POST"])
def login():
    """
    GET  — show the login page
    POST — check the submitted username and password
    """
    if is_logged_in():
        return redirect(url_for("home"))  # Already logged in, go to main page

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        users = load_users()

        # Check if username exists and password matches
        if username in users and users[username] == password:
            session["logged_in"] = True
            session["username"] = username  # Store username so we can show it in the nav bar
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid username or password.")

    return render_template("login.html")


@app.route("/logout")
def logout():
    """Clear the session and send the user back to the login page."""
    session.clear()
    return redirect(url_for("login"))


# ---- MAIN ROUTES (protected) ----

@app.route("/")
def home():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/screen", methods=["POST"])
def screen():
    if not is_logged_in():
        return redirect(url_for("login"))

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
        ai_name = extract_candidate_name(text, client)
        name = ai_name if ai_name else get_candidate_name(text, resume_file.filename)

        candidates.append({
            "name": name,
            "filename": resume_file.filename,
            "resume_text": text
        })

    # ---- STEP 4: SCREEN EACH CANDIDATE ----
    results = []

    for i, candidate in enumerate(candidates):
        if i > 0:
            time.sleep(3)

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
                "content": f"""Extract only the job title. One line, max 5 words, no punctuation.

JD:
{jd_text[:300]}"""
            }]
        )
        title = response.choices[0].message.content.strip()
        title = title.replace("\n", " ").replace("\r", " ")
        return title.split("\n")[0].strip()
    except:
        return "Role"


def build_prompt(name, resume_text, jd_text):
    """Compressed prompt — same output quality, minimal token usage."""
    resume_truncated = resume_text[:3000] if len(resume_text) > 3000 else resume_text
    jd_truncated = jd_text[:2000] if len(jd_text) > 2000 else jd_text

    return f"""You are an expert HR recruiter. Analyse the resume against the JD.
Return ONLY valid JSON — no markdown, no backticks, no explanation.

RULES:

1. FIT CATEGORY: exact_fit / semantic_fit / transferable_fit / irrelevant
   - exact_fit: meets all requirements directly
   - semantic_fit: same capability, different terminology
   - transferable_fit: different domain, skills genuinely transfer
   - irrelevant: no meaningful overlap

2. EXPERIENCE BAND: count full-time roles only, not internships or side projects.
   Add up sequential non-overlapping role durations to get total years.
   Do not count a role twice. If roles overlap, count the period once.
   - Early Career: 0-3 yrs | Mid Level: 3-8 yrs | Senior: 8-15 yrs | Leadership: 15+ yrs
   experience_band_reasoning: one short sentence stating total years counted and which roles.
   Format: "~X years total: [Role A] (Y yrs) + [Role B] (Z yrs)"

3. CURRENT ROLE: most recent substantive employed role at a named organisation.
   Ignore YouTube/creator/freelance/side projects unless only experience.
   Return NULL if not found.

4. MANDATORY REQUIREMENTS: only items JD explicitly marks as
   required/mandatory/must-have/essential. Empty list if none stated. Never infer.

5. STRENGTHS: interpreted takeaways with specific evidence.
   Format: "[capability] — evidenced by [specific project/metric/role]"
   Use consistent past tense: "evidenced by building X" not "evidenced by built X"
   No generic statements. Omit if no specific evidence exists.

6. GAPS: all shortcomings specific to this role. State why each matters.
   Flag missing mandatory requirements as gaps with higher priority.

7. HIGHLIGHTS: the 3-5 most impressive signals from this resume, priority order:
   a) Quantified achievements — metrics, percentages, scale
   b) Career spikes — promotions, high-profile projects, scale milestones
   c) Pedigree — top-tier institutions, well-known employers, notable brands
   Short punchy lines only. Do not repeat content from Strengths.
   If no highlights exist, return empty list.
   For irrelevant candidates, return empty list.

8. RED FLAGS — always return this field:
   Only flag: (a) 2+ roles under 12 months excluding contracts/internships,
   (b) unexplained gaps over 6 months, (c) clear overqualification,
   (d) skills claimed with zero supporting project evidence.
   Missing skills are NOT red flags. If none: ["No red flags identified."]

9. CANDIDATE SKILLS: JD-relevant skills only.
   If candidate has NO skills relevant to the JD, return empty list [].
   Do not list skills completely unrelated to the role.
   context: hands_on / oversight / exposure

10. INDUSTRIES AND DOMAINS: from company names and role descriptions only.
    2-4 items max. Be specific — not "Technology" but "B2B SaaS" or "Fintech".

11. INTERVIEW FOCUS AREAS:
    For irrelevant candidates: return []
    For all others: candidate-specific questions only.
    Each must reference a specific gap, claim, or inconsistency in THIS resume.
    BANNED — do not write any variation of:
    - "How do you stay up-to-date with new tools?"
    - "Describe a time you communicated with a non-technical audience"
    - "Walk me through your experience with X" (too broad)
    - "Can you give an example of a project you led?"
    GOOD examples:
    - "Your resume shows Looker but role needs Tableau — have you used Tableau,
       and what would your ramp-up look like?"
    - "You list Python as a skill but no project uses it — describe a specific
       script or analysis you built."
    - "You were promoted to Senior in 18 months — what drove that, and how
       does it translate here?"
    Min 3, max 5. All must be specific to this candidate.

12. RECOMMENDATION:
    Write 2-3 sentences from the recruiter to the hiring manager.
    Tone: candid and direct, as a recruiter speaking — not a formal report.

    Include:
    - Who the candidate is: current role, years, domain
    - The single most important reason to pursue or pass
    - A clear value judgement: worth a conversation or not

    Calibration:
    - 0-1 missing mandatory requirements → recommend with caveats
    - 2+ missing → flag gaps clearly, cautious tone
    - 3+ missing → not recommended, be direct
    - irrelevant fit → one sentence, do not recommend

    Do not mention next steps, interview rounds, or the screening tool.
    Do not use phrases like "strong communicator" without specific evidence.

13. EDUCATION SCORE:
    100% only if JD specifies an exact degree AND candidate has it exactly.
    80-90% if degree is relevant but not exactly specified.
    60-70% if tangentially related.
    Below 60% if unrelated or missing entirely.

JSON STRUCTURE:
{{
  "overall_match": <0-100>,
  "skills_match": <0-100>,
  "experience_match": <0-100>,
  "education_match": <0-100>,
  "experience_band": <"Early Career"|"Mid Level"|"Senior"|"Leadership">,
  "experience_years": "<e.g. '~4 years'>",
  "experience_band_reasoning": "<~X years total: Role A (Y yrs) + Role B (Z yrs)>",
  "current_title": "<title or NULL>",
  "current_company": "<company or NULL>",
  "industries_and_domains": ["<specific industry or domain>"],
  "fit_category": <"exact_fit"|"semantic_fit"|"transferable_fit"|"irrelevant">,
  "fit_reasoning": "<one sentence>",
  "score_reasoning": {{
    "skills": "<specific skills found or missing>",
    "experience": "<years and roles vs required>",
    "education": "<qualification found vs required>"
  }},
  "mandatory_requirements": [
    {{"requirement": "<text>", "met": <true|false>, "reason": "<one sentence>"}}
  ],
  "highlights": ["<punchy achievement, spike, or pedigree signal>"],
  "strengths": ["<capability — evidenced by specific detail>"],
  "gaps": ["<specific gap — why it matters>"],
  "candidate_skills": [
    {{"skill": "<name>", "context": <"hands_on"|"oversight"|"exposure">}}
  ],
  "red_flags": ["<specific flag or 'No red flags identified.'>"],
  "interview_focus_areas": ["<candidate-specific question or empty list>"],
  "recommendation": "<Shortlist for X / Reject / Hold.\\nOne sentence reason.>"
}}

JOB DESCRIPTION:
{jd_truncated}

RESUME — {name}:
{resume_truncated}"""


def screen_candidate(name, resume_text, jd_text):
    """Screen one candidate. Retries once on parse failure."""
    prompt = build_prompt(name, resume_text, jd_text)

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=4000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences if model wraps response
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            data = json.loads(raw)
            return data

        except json.JSONDecodeError:
            if attempt == 0:
                time.sleep(2)
                continue
            return _fallback("Could not parse AI response after retry")

        except Exception as e:
            error_str = str(e)
            if attempt == 0:
                time.sleep(2)
                continue
            if "429" in error_str or "rate_limit" in error_str.lower():
                return _fallback(
                    "Daily screening limit reached. "
                    "Results will resume after midnight IST, "
                    "or visit console.groq.com to upgrade your plan "
                    "and set a spend limit."
                )
            return _fallback(f"Error: {str(e)}")

    return _fallback("Analysis failed after retry")


def _fallback(reason):
    """Safe empty result when AI response fails."""
    return {
        "overall_match": 0,
        "skills_match": 0,
        "experience_match": 0,
        "education_match": 0,
        "experience_band": "Unknown",
        "experience_years": "Unknown",
        "experience_band_reasoning": reason,
        "current_title": "Unknown",
        "current_company": "Unknown",
        "industries_and_domains": [],
        "fit_category": "irrelevant",
        "fit_reasoning": reason,
        "score_reasoning": {
            "skills": "Could not parse",
            "experience": "Could not parse",
            "education": "Could not parse"
        },
        "mandatory_requirements": [],
        "highlights": [],
        "strengths": [],
        "gaps": [],
        "candidate_skills": [],
        "red_flags": [reason],
        "interview_focus_areas": [],
        "recommendation": "Analysis failed. Please try again."
    }


# ---- DOWNLOAD ROUTES (protected) ----

@app.route("/download-csv", methods=["POST"])
def download_csv():
    if not is_logged_in():
        return jsonify({"error": "Not logged in"}), 401

    """CSV with logical column ordering. New AI fields appended at end automatically."""
    data = request.get_json()
    results = data.get("results", [])
    job_title = data.get("job_title", "Role")
    job_title = job_title.replace("\n", " ").replace("\r", " ").strip()

    if not results:
        return jsonify({"error": "No results to download"}), 400

    def flatten(value):
        if isinstance(value, list):
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

    preferred_order = [
        "name", "current_title", "current_company", "filename",
        "experience_years", "experience_band", "industries_and_domains",
        "fit_category", "overall_match", "skills_match",
        "experience_match", "education_match", "recommendation",
        "mandatory_requirements",
        "highlights", "strengths", "gaps", "red_flags",
        "interview_focus_areas", "candidate_skills",
        "fit_reasoning", "experience_band_reasoning", "score_reasoning",
    ]

    all_result_keys = set()
    for r in results:
        all_result_keys.update(r.keys())

    ordered_keys = [k for k in preferred_order if k in all_result_keys]
    remaining = [k for k in all_result_keys if k not in preferred_order]
    final_keys = ordered_keys + sorted(remaining)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(final_keys)

    for r in results:
        writer.writerow([flatten(r.get(k, "")) for k in final_keys])

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


@app.route("/debug-users")
def debug_users():
    import os
    raw1 = os.getenv("APP_USERS", "NOT FOUND")
    raw2 = os.environ.get("APP_USERS", "NOT FOUND 2")
    return jsonify({
        "getenv": raw1[:5] if raw1 not in ["NOT FOUND"] else raw1,
        "environ": raw2[:5] if raw2 not in ["NOT FOUND 2"] else raw2,
        "all_keys": [k for k in os.environ.keys() if "USER" in k.upper()]
    })

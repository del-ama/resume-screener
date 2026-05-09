import PyPDF2
import docx
import io


def extract_text_from_pdf(file):
    """Extract text from a PDF file object"""
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def extract_text_from_docx(file):
    """Extract text from a Word (.docx) file object"""
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading Word file: {str(e)}"


def extract_text(file, filename):
    """
    Master function — detects file type by extension
    and routes to the correct extractor.
    """
    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif filename_lower.endswith(".docx"):
        return extract_text_from_docx(file)
    elif filename_lower.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return "Unsupported file type. Please upload PDF, DOCX, or TXT."


def get_candidate_name(text, filename):
    """
    Fallback name from filename.
    e.g. 'rahul_sharma_resume.pdf' → 'Rahul Sharma Resume'
    """
    name = filename.rsplit(".", 1)[0]
    name = name.replace("_", " ").replace("-", " ")
    return name.title()


def extract_candidate_name(resume_text, client):
    """
    Use AI to extract the candidate's actual name from resume text.
    Returns None if it can't find one — caller uses filename instead.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=20,
            temperature=0,
            messages=[{
                "role": "user",
                "content": f"""Extract only the candidate's full name from this resume.
Return just the name, nothing else. No punctuation, no explanation.
If you cannot find a name, return the word NULL.

Resume (first 500 characters):
{resume_text[:500]}"""
            }]
        )
        name = response.choices[0].message.content.strip()
        if name == "NULL" or len(name) > 50:
            return None
        return name
    except:
        return None
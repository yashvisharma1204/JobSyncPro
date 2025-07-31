from flask import Flask, request, render_template, session, redirect, url_for, jsonify
import os
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import Matcher, PhraseMatcher
import re
import logging
import google.generativeai as genai
import json
import concurrent.futures
import pickle
import hashlib
import uuid
import shutil
from functools import wraps
from datetime import timedelta

# --- Firebase Admin Setup ---
import firebase_admin
from firebase_admin import credentials, auth

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads/'
app.config['CACHE_FOLDER'] = 'Cache/'
app.config['SECRET_KEY'] = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)

# --- Initialize Firebase Admin SDK ---
try:
    # Get the absolute path to the directory where main.py is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Join that path with the filename to create a reliable path
    service_account_path = os.path.join(BASE_DIR, 'firebase-service-account.json')
    
    # Use the absolute path to initialize
    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(cred)
    logging.info("Firebase Admin SDK initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Firebase Admin SDK. Make sure 'firebase-service-account.json' is present. Error: {e}")
    # You might want to exit or handle this more gracefully depending on your needs
    # For now, the app will run but auth-dependent routes will fail.


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set. Please set it before running the application.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

#SpaCy Model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {str(e)}. Please run 'python -m spacy download en_core_web_sm'")
    raise

#  Skill Definitions (Consistently Lowercase)
SOFT_SKILLS = {
    "communication", "teamwork", "leadership", "problem-solving", "adaptability",
    "time management", "collaboration", "creativity", "work ethic", "interpersonal skills",
    "organizational skills", "critical thinking", "negotiation", "conflict resolution",
    "emotional intelligence", "decision making",
    "presentation skills", "active listening", "attention to detail", "analytical skills"
}

TECHNICAL_SKILLS = {
    "python", "java", "javascript", "html", "css", "mysql", "mongodb", "pandas",
    "numpy", "matplotlib", "scikit-learn", "power bi", "aws", "azure", "git",
    "github", "flask", "socketio", "pyspark", "databricks", "gemini api", "seaborn",
    "apache spark", "xgboost", "linear regression", "decision tree", "rest apis",
    "html/css", "ci/cd", "angular js", "react", "oracle", "machine learning",
    "deep learning", "natural language processing", "data analysis", "data visualization",
    "cloud computing", "devops", "kubernetes", "docker", "sql", "excel", "tableau",
    "r", "c++", "c#", "scala", "go", "spring boot", "node.js", "typescript",
    "data analysis and visualisation",
    "git and github",
    "rest apis",
    "apache spark",
    "linear regression",
    "decision tree",
    "aws lambda",
    "aws glue",
    "aws iam",
    "next.js",
    "react.js",
    "generative ai",
    "tkinter",
    "socket.io",
    "num py",
    "scikit learn"
}

# --- Login Required Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_uid' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function


# --- CORE LOGIC (UNCHANGED AS REQUESTED) ---

def clean_resume_text(resume_text):
    """Clean resume text to remove OCR artifacts and normalize formatting."""
    resume_text = re.sub(r'\b(\d+\s+)+\d+\b', ' ', resume_text)
    resume_text = re.sub(r'[^\x00-\x7F\u2000-\u206F\u20A0-\u20CF\u2100-\u214F\u2190-\u21FF\u2200-\u22FF\u2500-\u257F\u25A0-\u25FF\u2600-\u26FF\u2700-\u27BF\u2B00-\u2BFF\u2C60-\u2C7F\u2D00-\u2D2F\u2D30-\u2D6F\u2D80-\u2DDF\u2E00-\u2E7F\u2E80-\u2EFF\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u3100-\u312F\u3130-\u318F\u3190-\u31AF\u31C0-\u31EF\u3200-\u32FF\u3300-\u33FF\uFE30-\uFE4F\uFF00-\uFFEF\x20-\x7E]+', ' ', resume_text)
    resume_text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', resume_text)
    resume_text = re.sub(r'\b(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b', ' ', resume_text)
    resume_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', resume_text, flags=re.IGNORECASE)
    resume_text = re.sub(r'[•●■▪\u2022\u2023\u25CF\u25AA\u2043]', ' ', resume_text)
    resume_text = re.sub(r'\s+', ' ', resume_text.strip())
    return resume_text

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        logger.debug(f"Extracted text from PDF {file_path}: {text[:200]}...")
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF {file_path}: {str(e)}")
        return ""

def extract_text_from_docx(file_path):
    try:
        text = docx2txt.process(file_path)
        logger.debug(f"Extracted text from DOCX {file_path}: {text[:200]}...")
        return text
    except Exception as e:
        logger.error(f"Error extracting DOCX {file_path}: {str(e)}")
        return ""

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            logger.debug(f"Extracted text from TXT {file_path}: {text[:200]}...")
            return text
    except Exception as e:
        logger.error(f"Error extracting TXT {file_path}: {str(e)}")
        return ""

def extract_text(file_path):
    logger.debug(f"Processing file: {file_path}")
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        logger.error(f"Unsupported file format: {file_path}")
        return ""

#  Caching Functions
def get_gemini_response_cache_key(job_description, resume_text, prompt):
    combined_input = job_description + "|||" + resume_text + "|||" + prompt
    return hashlib.md5(combined_input.encode('utf-8')).hexdigest()

def cache_gemini_response(gemini_response, cache_key):
    cache_path = os.path.join(app.config['CACHE_FOLDER'], f"gemini_{cache_key}.pkl")
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(gemini_response, f)
        logger.debug(f"Cached Gemini response: {cache_key}")
    except Exception as e:
        logger.error(f"Error caching Gemini response {cache_key}: {str(e)}")

def load_cached_gemini_response(cache_key):
    cache_path = os.path.join(app.config['CACHE_FOLDER'], f"gemini_{cache_key}.pkl")
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        else:
            return None
    except Exception as e:
        logger.error(f"Error loading cached Gemini response {cache_key}: {str(e)}")
        return None

def get_resume_cache_key(resume_text):
    """Generate a cache key based on resume text hash."""
    return hashlib.md5(resume_text.encode('utf-8')).hexdigest()

def cache_resume(parsed_resume, cache_key):
    """Cache parsed resume to disk."""
    cache_path = os.path.join(app.config['CACHE_FOLDER'], f"parsed_resume_{cache_key}.pkl")
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(parsed_resume, f)
        logger.debug(f"Cached parsed resume: {cache_key}")
    except Exception as e:
        logger.error(f"Error caching parsed resume {cache_key}: {str(e)}")

def load_cached_resume(cache_key):
    """Load cached resume from disk."""
    cache_path = os.path.join(app.config['CACHE_FOLDER'], f"parsed_resume_{cache_key}.pkl")
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        else:
            return None
    except Exception as e:
        logger.error(f"Error loading cached parsed resume {cache_key}: {str(e)}")
        return None

#Gemini API Call Function
def get_gemini_response(job_description, resume_text, prompt):
    if not GEMINI_API_KEY:
        logger.warning("Gemini API key not set, skipping API call and returning default error.")
        return json.dumps({
            "percentage_match": 0.0,
            "missing_keywords": [{"keyword": "Gemini API key not configured.", "type": "system", "importance": "critical"}],
            "recommendations": ["Set GEMINI_API_KEY environment variable."],
            "quality_feedback": {}
        })

    generation_config = {
        "temperature": 0.0, # CRITICAL FOR CONSISTENCY
        "top_p": 1.0,
        "top_k": 1,
    }

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        input_text = f"Job Description:\n{job_description}\n\nResume:\n{resume_text}"

        response = model.generate_content([input_text, prompt], generation_config=generation_config)

        response_text = response.text
        response_text = re.sub(r'^```json\n|\n```$', '', response_text, flags=re.MULTILINE)
        logger.debug(f"Gemini API raw response: {response_text[:500]}...")
        return response_text
    except Exception as e:
        logger.error(f"Error in Gemini API response: {str(e)}")
        return json.dumps({
            "percentage_match": 0.0,
            "missing_keywords": [{"keyword": f"Gemini API call failed: {str(e)}", "type": "system", "importance": "critical"}],
            "recommendations": ["Check Gemini API key, network, and prompt structure."],
            "quality_feedback": {}
        })

# Resume Parsing Function
def parse_resume(resume_text):
    logger.debug(f"--- Starting parse_resume ---")
    logger.debug(f"Initial resume text received: {resume_text[:200]}...")

    resume_text = clean_resume_text(resume_text)
    logger.debug(f"Cleaned resume text: {resume_text[:200]}...")
    doc = nlp(resume_text)

    matcher = Matcher(nlp.vocab)
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    section_patterns = [
        [{"LOWER": {"IN": ["summary", "about me", "profile"]}}],
        [{"LOWER": {"IN": ["skills", "technical skills", "core competencies", "extracurricular skills", "technologies", "tech skills"]}}],
        [{"LOWER": {"IN": ["experience", "work experience", "professional experience"]}}],
        [{"LOWER": "education"}],
        [{"LOWER": "projects"}],
        [{"LOWER": {"IN": ["achievements", "certifications", "additional information", "awards", "languages", "publications"]}}]
    ]
    matcher.add("SECTION", section_patterns)

    soft_skill_phrases = [nlp.make_doc(phrase) for phrase in SOFT_SKILLS]
    phrase_matcher.add("SOFT_SKILLS", soft_skill_phrases)

    sections_raw_lines = {
        "summary": [], "skills": [], "experience": [],
        "education": [], "projects": [], "additional_info": []
    }
    current_section = None

    lines = [line.strip() for line in resume_text.split('\n') if line.strip()]

    for line_idx, line in enumerate(lines):
        line_doc = nlp(line)
        line_matches = matcher(line_doc)

        found_section_in_line = False
        for match_id, start, end in line_matches:
            section_text = line_doc[start:end].text.lower()
            if any(s in section_text for s in ["summary", "about me", "profile"]):
                current_section = "summary"
            elif any(s in section_text for s in ["skills", "technical skills", "core competencies", "extracurricular skills", "technologies", "tech skills"]):
                current_section = "skills"
            elif "experience" in section_text:
                current_section = "experience"
            elif "education" in section_text:
                current_section = "education"
            elif "projects" in section_text:
                current_section = "projects"
            elif any(s in section_text for s in ["achievements", "certifications", "additional information", "awards", "languages", "publications"]):
                current_section = "additional_info"
            else:
                continue
            found_section_in_line = True
            break

        if not found_section_in_line and current_section:
            sections_raw_lines[current_section].append(line)
        elif not found_section_in_line and not current_section:
            sections_raw_lines["summary"].append(line)

    skills = []
    soft_skills = []

    if sections_raw_lines["skills"]:
        skills_text_from_section = " ".join(sections_raw_lines["skills"]).lower()
        tech_matcher_for_section = PhraseMatcher(nlp.vocab, attr="LOWER")
        tech_phrases_for_section = [nlp.make_doc(s) for s in TECHNICAL_SKILLS]
        tech_matcher_for_section.add("EXACT_TECH_SKILLS_SECTION", tech_phrases_for_section)

        skills_doc_section = nlp(skills_text_from_section)
        for match_id, start, end in tech_matcher_for_section(skills_doc_section):
            matched_skill = skills_doc_section[start:end].text.lower()
            if matched_skill not in skills:
                skills.append(matched_skill)

        words_in_skills_section = re.findall(r'\b[a-zA-Z0-9+#.-]+\b', skills_text_from_section)
        for word in words_in_skills_section:
            word_lower = word.strip().lower()
            if word_lower in TECHNICAL_SKILLS and word_lower not in skills:
                skills.append(word_lower)

    full_resume_doc = nlp(resume_text)
    all_tech_phrases = [nlp.make_doc(s) for s in TECHNICAL_SKILLS]
    phrase_matcher.add("ALL_TECHNICAL_SKILLS_GLOBAL", all_tech_phrases)

    for match_id, start, end in phrase_matcher(full_resume_doc):
        if nlp.vocab.strings[match_id] == "ALL_TECHNICAL_SKILLS_GLOBAL":
            matched_skill = full_resume_doc[start:end].text.lower()
            if matched_skill not in skills:
                skills.append(matched_skill)

    for token in full_resume_doc:
        token_text = token.text.lower()
        if token_text in TECHNICAL_SKILLS and token_text not in skills:
            skills.append(token_text)

    for match_id, start, end in phrase_matcher(doc):
        if nlp.vocab.strings[match_id] == "SOFT_SKILLS":
            matched_soft_skill = doc[start:end].text.lower()
            if matched_soft_skill not in soft_skills:
                soft_skills.append(matched_soft_skill)

    skills = sorted(list(set(skills)))
    soft_skills = sorted(list(set(soft_skills)))

    sections_content_dict = {k: " ".join(v) for k, v in sections_raw_lines.items()}

    parsed_data = {
        "skills": skills,
        "soft_skills": soft_skills,
        "raw_text": resume_text,
        "sections": sections_content_dict
    }

    logger.debug(f"--- Finished parse_resume ---")
    logger.debug(f"Parsed resume FINAL technical skills: {parsed_data['skills']}")
    logger.debug(f"Parsed resume FINAL soft skills: {parsed_data['soft_skills']}")
    return parsed_data

def analyze_resume_quality(resume_text, sections_content_dict):
    quality_feedback = {
        "grammar_issues": [], # This will primarily be populated by Gemini
        "structural_suggestions": [],
        "conciseness_suggestions": [],
    }
    quality_feedback["grammar_issues"].append("Grammar and spelling analysis is primarily provided by the AI (Gemini) for advanced contextual review.")
    detected_sections = list(sections_content_dict.keys())
    standard_sections = ["summary", "experience", "education", "skills", "projects"]
    missing_standard = [s for s in standard_sections if s not in detected_sections or not sections_content_dict.get(s, "").strip()]
    if missing_standard:
        quality_feedback["structural_suggestions"].append(
            f"Consider ensuring your resume includes standard sections like: {', '.join(missing_standard).title()}."
        )
    for section_name in ["experience", "projects"]:
        if sections_content_dict.get(section_name):
            section_content = sections_content_dict[section_name]
            lines = [line.strip() for line in section_content.split('\n') if line.strip()]
            bullet_start_count = 0
            for line in lines:
                if re.match(r'^\s*[\u2022\u2023\u25CF\u25AA\u2043*-]', line):
                    bullet_start_count += 1
            if len(lines) > 5 and bullet_start_count < len(lines) * 0.7:
                quality_feedback["structural_suggestions"].append(
                    f"In your '{section_name.title()}' section, consider using more consistent bullet points for readability and ATS parsing."
                )
    if not quality_feedback["structural_suggestions"]:
        quality_feedback["structural_suggestions"].append("Basic resume structure appears well-organized for ATS compatibility.")
    word_count = len(resume_text.split())
    estimated_pages = max(1, round(word_count / 500))
    if word_count > 1000 and estimated_pages > 2:
        quality_feedback["conciseness_suggestions"].append(
            f"Your resume has approximately {estimated_pages} pages ({word_count} words). For many roles, a concise 1-2 page resume is preferred. Consider condensing information."
        )
    elif word_count > 500 and estimated_pages > 1 and len(sections_content_dict.get("experience", "").split('\n')) < 5:
        quality_feedback["conciseness_suggestions"].append(
            f"Your resume has approximately {estimated_pages} pages ({word_count} words). For early career professionals, a single-page resume is often ideal. Review for conciseness."
        )
    if not quality_feedback["conciseness_suggestions"]:
        quality_feedback["conciseness_suggestions"].append("Resume length appears appropriate.")
    return quality_feedback

def process_resume(resume_file, job_description, input_prompt):
    """Process a single resume file."""
    if not resume_file or not resume_file.filename:
        logger.warning("Empty or no resume file received in process_resume.")
        return None, 0.0, {"percentage_match": 0.0, "missing_keywords": [], "recommendations": [], "quality_feedback": {}}, {"skills": [], "soft_skills": [], "raw_text": ""}, ""

    # This function is now called with a MockFileStorage object which has a `filepath` attribute
    # The original save logic is now handled by MockFileStorage.save(), which just copies the file.
    filename_for_processing = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_processing', resume_file.filename)
    try:
        # Create a temporary subdir for this file to avoid name clashes in concurrent processing
        temp_processing_dir = os.path.dirname(filename_for_processing)
        os.makedirs(temp_processing_dir, exist_ok=True)
        resume_file.save(filename_for_processing) # This now copies from job_upload_path to temp_processing_path
        logger.debug(f"Copied resume file for processing: {filename_for_processing}")


        resume_text = extract_text(filename_for_processing)
        if not resume_text.strip():
            logger.error(f"Failed to extract text from {filename_for_processing} (empty content).")
            return None, 0.0, {"percentage_match": 0.0, "missing_keywords": [], "recommendations": [], "quality_feedback": {}}, {"skills": [], "soft_skills": [], "raw_text": ""}, ""

        parsed_resume_cache_key = get_resume_cache_key(resume_text)
        parsed_resume = load_cached_resume(parsed_resume_cache_key)

        if parsed_resume:
            logger.debug(f"Loaded parsed resume from cache for {resume_file.filename}.")
            if "sections" not in parsed_resume or not isinstance(parsed_resume["sections"], dict):
                logger.warning(f"Cached resume for {resume_file.filename} missing 'sections' key or it's malformed. Re-parsing.")
                parsed_resume = parse_resume(resume_text)
                cache_resume(parsed_resume, parsed_resume_cache_key)
        else:
            parsed_resume = parse_resume(resume_text)
            cache_resume(parsed_resume, parsed_resume_cache_key)
            logger.debug(f"Parsed and cached resume for {resume_file.filename}.")

        local_quality_feedback = analyze_resume_quality(parsed_resume["raw_text"], parsed_resume["sections"])

        gemini_response_cache_key = get_gemini_response_cache_key(job_description, resume_text, input_prompt)
        gemini_response_raw = load_cached_gemini_response(gemini_response_cache_key)

        if gemini_response_raw:
            logger.debug(f"Loaded Gemini response from cache for {resume_file.filename}.")
        else:
            logger.info(f"Calling Gemini API for {resume_file.filename}...")
            gemini_response_raw = get_gemini_response(job_description, resume_text, input_prompt)
            cache_gemini_response(gemini_response_raw, gemini_response_cache_key)
            logger.debug(f"Cached new Gemini response for {resume_file.filename}.")

        percentage = 0.0
        recommendation_data = {
            "percentage_match": 0.0,
            "missing_keywords": [],
            "recommendations": [],
            "quality_feedback": {}
        }

        try:
            response_json = json.loads(gemini_response_raw)
            percentage = float(response_json.get("percentage_match", 0.0))
            recommendation_data["percentage_match"] = percentage
            recommendation_data["missing_keywords"] = response_json.get("missing_keywords", [])
            recommendation_data["recommendations"] = response_json.get("recommendations", [])
            recommendation_data["quality_feedback"] = response_json.get("quality_feedback", {})
            logger.debug("Successfully parsed Gemini JSON response.")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for Gemini response: {e}. Raw response: {gemini_response_raw[:500]}...")
            match = re.search(r'"percentage_match"\s*:\s*(\d+\.?\d*)', gemini_response_raw)
            if match:
                percentage = float(match.group(1))
                recommendation_data["percentage_match"] = percentage
                recommendation_data["recommendations"].append(f"Warning: Gemini response JSON malformed. Percentage extracted by regex: {percentage}%. Raw response saved below.")
            else:
                percentage = 0.0
                recommendation_data["recommendations"].append(f"Error: Gemini response JSON malformed, and percentage not found via regex. Raw response saved below.")
            missing_kw_match = re.search(r'"missing_keywords"\s*:\s*\[(.*?)\]', gemini_response_raw, re.DOTALL)
            if missing_kw_match:
                try:
                    keywords_str = missing_kw_match.group(1).strip()
                    temp_list = []
                    try:
                        temp_list = json.loads(f"[{keywords_str}]")
                        if all(isinstance(item, dict) and "keyword" in item for item in temp_list):
                            recommendation_data["missing_keywords"] = temp_list
                        else:
                            raise ValueError("Not a list of keyword dicts")
                    except (json.JSONDecodeError, ValueError):
                        if keywords_str:
                            missing_keywords_list = [kw.strip().strip('"') for kw in re.split(r',\s*(?="|\b)', keywords_str) if kw.strip() and not kw.strip().startswith('{') and not kw.strip().endswith('}')]
                            keyword_objects = re.findall(r'{"keyword":"(.*?)"(?:,"type":"(.*?)"(?:,"importance":"(.*?)"|)|)}', keywords_str)
                            if keyword_objects:
                                for kw, type_val, imp_val in keyword_objects:
                                    item = {"keyword": kw}
                                    if type_val: item["type"] = type_val
                                    if imp_val: item["importance"] = imp_val
                                    if item not in recommendation_data["missing_keywords"]:
                                        recommendation_data["missing_keywords"].append(item)
                            elif missing_keywords_list:
                                recommendation_data["missing_keywords"].extend([
                                    {"keyword": k, "type": "unknown", "importance": "unknown"} for k in missing_keywords_list
                                    if {"keyword": k, "type": "unknown", "importance": "unknown"} not in recommendation_data["missing_keywords"]
                                ])
                except Exception as ex:
                    logger.debug(f"Failed to extract keywords from malformed JSON string '{keywords_str}' using fallback: {ex}")
                    pass
            quality_feedback_match = re.search(r'"quality_feedback"\s*:\s*({.*?})', gemini_response_raw, re.DOTALL)
            if quality_feedback_match:
                try:
                    quality_feedback_str = quality_feedback_match.group(1)
                    recommendation_data["quality_feedback"] = json.loads(quality_feedback_str)
                except json.JSONDecodeError as ex:
                    logger.debug(f"Failed to extract quality_feedback from malformed JSON: {ex}")
                    recommendation_data["quality_feedback"] = {"parsing_error": f"Failed to parse quality_feedback: {str(ex)}"}
            recommendation_data["recommendations"].append(f"Full Raw Gemini Response (for debugging): {gemini_response_raw}")
        final_quality_feedback = {}
        final_quality_feedback.update(local_quality_feedback)
        final_quality_feedback.update(recommendation_data["quality_feedback"])
        recommendation_data["quality_feedback"] = final_quality_feedback

        # Cleanup the temp file for this specific process
        try:
            os.remove(filename_for_processing)
            logger.debug(f"Removed temporary processing file: {filename_for_processing}")
        except OSError as e:
            logger.error(f"Error removing file {filename_for_processing}: {e}")

        return resume_file.filename, percentage, recommendation_data, parsed_resume, resume_text
    except Exception as e:
        logger.error(f"Critical error processing file {resume_file.filename}: {str(e)}")
        try:
            if os.path.exists(filename_for_processing):
                os.remove(filename_for_processing)
        except OSError:
            pass
        error_recommendation_data = {
            "percentage_match": 0.0,
            "missing_keywords": [],
            "recommendations": [f"Critical error processing {resume_file.filename}: {str(e)}", "Please check server logs for details."],
            "quality_feedback": {"system_error": f"Critical processing error: {str(e)}"}
        }
        return None, 0.0, error_recommendation_data, {"skills": [], "soft_skills": [], "raw_text": ""}, ""


# --- Internal Helper Class for New Workflow ---
class MockFileStorage:
    """A helper class to mimic a Flask FileStorage object from a file path."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)

    def save(self, dst):
        """Copies the file to the destination, mimicking the save method."""
        try:
            # The original process_resume expects to save the file itself.
            shutil.copy(self.filepath, dst)
        except Exception as e:
            logger.error(f"MockFileStorage could not copy {self.filepath} to {dst}: {e}")
            raise


# --- Flask Routes (UPDATED WITH AUTH) ---

@app.route("/")
def index():
    """
    Serves the public landing page (main.html). No login is required.
    """
    return render_template('main.html')

@app.route("/login")
def login_page():
    """
    Serves the authentication page. If the user is already logged in,
    it redirects them straight to the resume matcher tool.
    """
    if 'user_uid' in session:
        return redirect(url_for('home_page'))
    return render_template('auth.html')

@app.route("/home")
@login_required
def home_page():
    """
    This is the main tool page (uploader form), protected by the login_required decorator.
    """
    session.pop('job_id', None)
    session.pop('job_description', None)
    session.pop('resume_filenames', None)
    return render_template('home.html')

@app.route("/logout")
def logout():
    """
    Logs the user out and redirects to the public landing page.
    """
    session.clear()
    return redirect(url_for('index'))

@app.route('/verify-token', methods=['POST'])
def verify_token():
    """
    Verifies the Firebase token. The client-side JS in auth.html will
    redirect to '/home' on success.
    """
    try:
        token = request.json['token']
        decoded_token = auth.verify_id_token(token)
        session['user_uid'] = decoded_token['uid']
        session['user_email'] = decoded_token.get('email', 'N/A')
        session.permanent = True
        logger.info(f"User authenticated: {session['user_email']} ({session['user_uid']})")
        return jsonify({'status': 'success', 'uid': decoded_token['uid']}), 200
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        return jsonify({'status': 'error', 'message': 'Invalid token'}), 401

@app.route('/matcher', methods=['POST'])
@login_required
def loader():
    """
    This route is unchanged. It handles the form submission from home.html.
    """
    job_description = request.form.get('job_description', '').strip()
    resume_files = request.files.getlist('resumes')

    if not job_description or not resume_files or not any(f.filename for f in resume_files):
        logger.warning("Loader: Missing job description or resume files.")
        return render_template('home.html', message="Please provide a job description and at least one resume.")

    job_id = str(uuid.uuid4())
    job_upload_path = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    os.makedirs(job_upload_path, exist_ok=True)

    saved_filenames = []
    for f in resume_files:
        if f.filename:
            f.save(os.path.join(job_upload_path, f.filename))
            saved_filenames.append(f.filename)

    session['job_id'] = job_id
    session['job_description'] = job_description
    session['resume_filenames'] = saved_filenames

    logger.info(f"Job {job_id} initiated for user {session.get('user_email')}. Displaying loader page.")
    return render_template('loader.html')

@app.route('/results')
@login_required
def results():
    """
    This route is mostly unchanged. It processes and shows the results.
    The error redirect now goes back to the public landing page.
    """
    job_id = session.get('job_id')
    job_description = session.get('job_description')
    resume_filenames = session.get('resume_filenames')

    if not all([job_id, job_description, resume_filenames]):
        logger.error("Results page accessed without job data in session. Redirecting to home.")
        return redirect(url_for('index')) # Redirect to public home on error

    job_upload_path = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    resume_files = [MockFileStorage(os.path.join(job_upload_path, fname)) for fname in resume_filenames]

    logger.info(f"Processing job {job_id} retrieved from session with {len(resume_files)} resumes.")

    processed_results = []
    input_prompt = """
    You are a skilled ATS (Applicant Tracking System) scanner and career coach with a deep understanding of data science, resume best practices, and ATS functionality.
    Evaluate the resume against the provided job description. Your goal is to provide a highly accurate and granular percentage match,
    along with detailed missing keywords, actionable recommendations for improvement, and comprehensive textual feedback on resume quality.

    **Instructions for Percentage Match (percentage_match):**
    When calculating 'percentage_match', factor in ALL aspects of a strong resume for the given job. This includes:
    - **Technical Skills (e.g., programming languages, frameworks, tools, algorithms):** 50% contribution to this score. Give **very high credit** for skills **demonstrated in projects or experience descriptions**, even if not explicitly listed in a 'skills' section. Assess the depth of experience with *demonstrated* technologies. Focus heavily on the core requirements identified in the JD; **do not penalize significantly** for the absence of technologies listed as "such as" or in comprehensive lists that represent optional or alternative technologies.
    - **Work Experience & Projects (relevance, depth, impact, quantifiable achievements, duration):** 30% contribution to this score. Look for direct relevance to the job duties and give **maximum possible weight** to projects that directly simulate or align with aspects of the job description (e.g., full-stack development, API design, database management). Emphasize **strong, quantifiable results and clear impact**.
    - **Soft Skills (e.g., communication, problem-solving, teamwork, leadership):** 10% contribution to this score.
    - **Education & Certifications (relevance of degree, institutions, relevant courses):** 10% contribution to this score.
    - **Resume Quality (Grammar, Structure, Conciseness, Professionalism):** This is integrated into the overall 'percentage_match'. A well-written, error-free, and well-structured resume that is concise and professional should positively impact this score. **Specifically, if the resume shows good grammar, clear structure, and effective conciseness, significantly boost the overall 'percentage_match'. Conversely, any significant issues in these areas should noticeably reduce the score.**

    For each missing keyword, specify if it's a critical technical skill, a key soft skill, or a general requirement. Provide an 'importance' level: "critical", "important", or "optional".

    **Instructions for Detailed Resume Quality Feedback (quality_feedback):**
    Provide detailed *textual* feedback for the candidate to improve their resume's quality, even if the 'percentage_match' is high. This feedback should be actionable and specific.
    - **Grammar & Spelling:** Identify and list specific instances of grammatical errors, typos, punctuation mistakes, and awkward phrasing. If no significant errors, state that or mention the resume appears well-written. Focus on high-impact errors.
    - **Resume Structure & Formatting:** Assess if the resume follows standard, ATS-friendly formatting (e.g., clear, consistent headings like 'Work Experience', 'Education', 'Skills', 'Projects'; proper use of bullet points; avoidance of excessive graphics). Provide concrete suggestions for structural improvements.
    - **Quantifiable Achievements:** Identify opportunities where the candidate could add quantifiable results or metrics (numbers, percentages, currency, scale) to demonstrate impact more effectively in their experience and project descriptions. Provide examples of how they might rephrase points to include impact.
    - **Action Verbs:** Note if the resume predominantly uses passive voice or weak verbs. Suggest areas where stronger, more impactful action verbs could be used to start bullet points or describe responsibilities (e.g., "managed," "developed," "implemented," "achieved").
    - **Conciseness:** Evaluate if the resume's length is appropriate for the candidate's experience level (e.g., 1 page for early career, 1-2 pages for mid-career). Suggest areas for condensation if it appears too long, or expansion if too brief.

    Return the entire response in a single JSON format. Ensure all keys and string values are properly quoted.
    ```json
    {
      "percentage_match": <number between 0 and 100, float, precise to one decimal place, e.g., 82.5>,
      "missing_keywords": [
        {"keyword": "<string>", "type": "technical/soft/general", "importance": "critical/important/optional"},
        ...
      ],
      "recommendations": [<string>, ...],
      "quality_feedback": {
        "grammar_issues": [<string>, ...],
        "structural_suggestions": [<string>, ...],
        "quantifiable_achievements": [<string>, ...],
        "action_verbs": [<string>, ...],
        "conciseness_suggestions": [<string>, ...]
      }
    }
    ```
    """

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_resume_file = {
            executor.submit(process_resume, resume_file, job_description, input_prompt): resume_file
            for resume_file in resume_files
        }
        for future in concurrent.futures.as_completed(future_to_resume_file):
            original_file = future_to_resume_file[future]
            filename, percentage, recommendation_data, parsed_resume, resume_text = future.result()

            if filename:
                processed_results.append({
                    "filename": filename,
                    "percentage_gemini": percentage,
                    "recommendation_data": recommendation_data,
                    "parsed_resume": parsed_resume,
                    "resume_text": resume_text
                })
            else:
                logger.error(f"Failed to process {original_file.filename}. Skipping this resume.")
                processed_results.append({
                    "filename": original_file.filename or "Unnamed_Resume",
                    "percentage_gemini": 0.0,
                    "recommendation_data": recommendation_data,
                    "parsed_resume": {"skills": [], "soft_skills": [], "raw_text": ""},
                    "resume_text": ""
                })

    if not processed_results or all(res["percentage_gemini"] == 0.0 and not res["resume_text"].strip() for res in processed_results):
        logger.warning("No valid resumes processed after all attempts.")
        return render_template('home.html', message="No valid resumes processed. Please check file formats or content.")

    final_resumes_for_display = []

    GEMINI_COMPREHENSIVE_WEIGHT = 0.85
    TFIDF_WEIGHT = 0.15

    for i, res in enumerate(processed_results):
        gemini_match_score = res["percentage_gemini"]

        tfidf_score = 0.0
        if res["resume_text"].strip() and job_description.strip():
            try:
                vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.9)
                vectors = vectorizer.fit_transform([job_description, res["resume_text"]])

                if vectors.shape[1] > 0:
                    similarity = cosine_similarity(vectors[0], vectors[1]).flatten()[0]
                    tfidf_score = round(similarity * 100, 2)
                    logger.debug(f"TF-IDF score for {res['filename']}: {tfidf_score}%")
                else:
                    logger.debug(f"TF-IDF vectorization resulted in empty features for {res['filename']}.")
            except Exception as e:
                logger.error(f"Error during TF-IDF calculation for {res['filename']}: {str(e)}")

        combined_score = (gemini_match_score * GEMINI_COMPREHENSIVE_WEIGHT) + (tfidf_score * TFIDF_WEIGHT)

        final_resumes_for_display.append({
            "filename": res["filename"],
            "score": round(combined_score, 2),
            "skills": res["parsed_resume"]["skills"],
            "soft_skills": res["parsed_resume"]["soft_skills"],
            "recommendation_data": res["recommendation_data"]
        })

    top_resumes = sorted(final_resumes_for_display, key=lambda x: x['score'], reverse=True)[:5]
    logger.info(f"Top matching resumes calculated for job {job_id}: {[r['filename'] for r in top_resumes]}")

    # --- Cleanup ---
    try:
        shutil.rmtree(job_upload_path)
        logger.info(f"Cleaned up temporary directory: {job_upload_path}")
    except OSError as e:
        logger.error(f"Error removing temporary directory {job_upload_path}: {e}")
    
    # Clear job-specific session data but keep user login
    session.pop('job_id', None)
    session.pop('job_description', None)
    session.pop('resume_filenames', None)

    return render_template(
        'ats.html',
        job_description=job_description,
        top_resumes=top_resumes
    )


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        logger.info(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")
    if not os.path.exists(app.config['CACHE_FOLDER']):
        os.makedirs(app.config['CACHE_FOLDER'])
        logger.info(f"Created cache folder: {app.config['CACHE_FOLDER']}")

    app.run(debug=True, host='0.0.0.0')
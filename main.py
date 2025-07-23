from flask import Flask, request, render_template
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
# Removed: from textblob import TextBlob

# --- Flask App Configuration ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads/'
app.config['CACHE_FOLDER'] = 'Cache/'

# --- Logging Setup ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Gemini API Key Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set. Please set it before running the application.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# --- SpaCy Model Loading ---
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {str(e)}. Please run 'python -m spacy download en_core_web_sm'")
    raise

# --- Skill Definitions (Consistently Lowercase) ---
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

# --- Text Extraction & Cleaning Functions ---
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

# --- Caching Functions ---
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

# --- Gemini API Call Function ---
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
        model = genai.GenerativeModel('gemini-2.0-flash')
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

# --- Resume Parsing Function ---
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

# --- Resume Quality Analysis Function (UPDATED - Removed TextBlob usage) ---
def analyze_resume_quality(resume_text, sections_content_dict):
    quality_feedback = {
        "grammar_issues": [], # This will now primarily be populated by Gemini
        "structural_suggestions": [],
        "conciseness_suggestions": [],
    }

    # 1. Grammar and Spelling Check - REMOVED TextBlob usage.
    # This part will now explicitly rely on Gemini's feedback.
    # If Gemini's response is missing this category, it will default to an empty list.
    # We can add a placeholder message if desired, but letting Gemini handle it fully is cleaner.
    quality_feedback["grammar_issues"].append("Grammar and spelling analysis is provided by the AI (Gemini) for advanced contextual review.")


    # 2. Structure Analysis (Rule-based)
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

    # 3. Conciseness Check
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

# --- Main Resume Processing Logic ---
def process_resume(resume_file, job_description, input_prompt):
    """Process a single resume file."""
    if not resume_file or not resume_file.filename:
        logger.warning("Empty or no resume file received in process_resume.")
        return None, 0.0, {"percentage_match": 0.0, "missing_keywords": [], "recommendations": [], "quality_feedback": {}}, {"skills": [], "soft_skills": [], "raw_text": ""}, ""

    filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        resume_file.save(filename)
        logger.debug(f"Saved resume file: {filename}")

        resume_text = extract_text(filename)
        if not resume_text.strip():
            logger.error(f"Failed to extract text from {filename} (empty content).")
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

        # Perform local quality analysis (now only structural/conciseness)
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

        # Combine local quality feedback with Gemini's quality feedback
        # Now, local_quality_feedback contains only structural/conciseness info
        # Gemini's quality_feedback will contain grammar, action verbs, quantifiable achievements, etc.
        # We merge them, preferring Gemini's if a category overlaps (e.g., if Gemini also gives structural tips)
        final_quality_feedback = {}
        # Start with local insights (structural, conciseness)
        final_quality_feedback.update(local_quality_feedback)
        # Overlay Gemini's insights, which will typically include grammar/spelling, action verbs, quantifiable.
        # This update ensures Gemini's more detailed feedback takes precedence if both provide a category.
        final_quality_feedback.update(recommendation_data["quality_feedback"])
        recommendation_data["quality_feedback"] = final_quality_feedback

        try:
            os.remove(filename)
            logger.debug(f"Removed temporary file: {filename}")
        except OSError as e:
            logger.error(f"Error removing file {filename}: {e}")

        return resume_file.filename, percentage, recommendation_data, parsed_resume, resume_text
    except Exception as e:
        logger.error(f"Critical error processing file {resume_file.filename}: {str(e)}")
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except OSError:
            pass
        error_recommendation_data = {
            "percentage_match": 0.0,
            "missing_keywords": [],
            "recommendations": [f"Critical error processing {resume_file.filename}: {str(e)}", "Please check server logs for details."],
            "quality_feedback": {"system_error": f"Critical processing error: {str(e)}"}
        }
        return None, 0.0, error_recommendation_data, {"skills": [], "soft_skills": [], "raw_text": ""}, ""

# --- Flask Routes ---
@app.route("/")
def matchresume():
    return render_template('main.html')

@app.route("/matchresume")
def matchr():
    return render_template('home.html')

@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form.get('job_description', '').strip()
        resume_files = request.files.getlist('resumes')

        logger.info(f"Received {len(resume_files)} resume files for matching.")
        logger.debug(f"Job description: {job_description[:200]}...")

        if not resume_files or not job_description:
            logger.warning("No resumes or job description provided by user.")
            return render_template('home.html', message="Please upload resumes and enter a job description.")

        processed_results = []

        input_prompt = """
        You are a skilled ATS (Applicant Tracking System) scanner and career coach with a deep understanding of data science, resume best practices, and ATS functionality.
        Evaluate the resume against the provided job description. Your goal is to provide a highly accurate and granular percentage match,
        along with detailed missing keywords, actionable recommendations for improvement, and comprehensive feedback on resume quality (grammar, structure, impact, conciseness).

        **Instructions for a more flexible evaluation (Match Score & Missing Keywords):**
        - **Prioritize core concepts and demonstrated experience** even if the exact keywords are not present. Look for evidence of the skill or experience through descriptions of projects, responsibilities, and achievements.
        - **Consider common synonyms and related terms** for technical and soft skills. For example, "data manipulation" for "data wrangling," or "collaborative" for "teamwork."
        - **Focus on the overall spirit of the requirement** rather than just a literal keyword match.
        - **Be generous in attributing a skill** if there's clear evidence of its application, even if the skill name isn't explicitly listed in a "Skills" section.
        - **Acknowledge and give credit for relevant foundational knowledge** even if advanced versions of a skill are missing.
        - **For "missing keywords," focus on truly absent critical requirements.** If a skill is present but could be elaborated, suggest elaboration in recommendations rather not listing it as "missing."

        Strictly adhere to the following percentage calculation weighting:
        - Technical Skills (e.g., programming languages, frameworks, tools, algorithms): 50%
        - Work Experience & Projects (relevance, depth, impact, quantifiable achievements, duration): 30%
        - Soft Skills (e.g., communication, problem-solving, teamwork, leadership): 10%
        - Education & Certifications (relevance of degree, institutions, relevant courses): 10%

        For each missing keyword, specify if it's a critical technical skill, a key soft skill, or a general requirement. Provide an 'importance' level: "critical", "important", or "optional".

        **Instructions for Resume Quality Feedback (beyond matching):**
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

        all_texts_for_tfidf_input = []
        original_indices_map = []

        if job_description.strip():
            all_texts_for_tfidf_input.append(job_description)
            job_desc_tfidf_idx = 0
        else:
            job_desc_tfidf_idx = -1

        for i, res in enumerate(processed_results):
            if res["resume_text"].strip():
                all_texts_for_tfidf_input.append(res["resume_text"])
                original_indices_map.append(i)

        tfidf_scores = [0.0] * len(processed_results)

        try:
            if len(all_texts_for_tfidf_input) >= 2:
                vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.9)
                vectors = vectorizer.fit_transform(all_texts_for_tfidf_input)

                if job_desc_tfidf_idx != -1 and vectors.shape[1] > 0:
                    job_vector = vectors[job_desc_tfidf_idx]

                    resume_vectors_for_tfidf = vectors[job_desc_tfidf_idx + 1:]

                    if resume_vectors_for_tfidf.shape[0] > 0:
                        similarities = cosine_similarity(job_vector, resume_vectors_for_tfidf).flatten()

                        for i, score in enumerate(similarities):
                            original_res_idx = original_indices_map[i]
                            tfidf_scores[original_res_idx] = round(score * 100, 2)
                    else:
                        logger.warning("No valid resume vectors for TF-IDF similarity calculation after filtering.")
                else:
                    logger.warning("TF-IDF vectorization resulted in empty features or job description not available for vectorization.")
            else:
                logger.warning("Insufficient valid texts for TF-IDF vectorization (need job description and at least one resume).")
        except Exception as e:
            logger.error(f"Error during TF-IDF vectorization: {str(e)}")


        final_resumes_for_display = []
        GEMINI_WEIGHT = 0.7
        TFIDF_WEIGHT = 0.3

        for i, res in enumerate(processed_results):
            gemini_score = res["percentage_gemini"]
            tfidf_score = tfidf_scores[i]

            combined_score = 0.0
            if gemini_score > 0.0 and tfidf_score > 0.0:
                combined_score = (gemini_score * GEMINI_WEIGHT) + (tfidf_score * TFIDF_WEIGHT)
            elif gemini_score > 0.0:
                combined_score = gemini_score
            elif tfidf_score > 0.0:
                combined_score = tfidf_score

            final_resumes_for_display.append({
                "filename": res["filename"],
                "score": round(combined_score, 2),
                "skills": res["parsed_resume"]["skills"],
                "soft_skills": res["parsed_resume"]["soft_skills"],
                "recommendation_data": res["recommendation_data"]
            })

        top_resumes = sorted(final_resumes_for_display, key=lambda x: x['score'], reverse=True)[:5]

        logger.info(f"Top matching resumes calculated: {[r['filename'] for r in top_resumes]}")
        return render_template(
            'ats.html',
            job_description=job_description,
            top_resumes=top_resumes
        )

    return render_template('home.html')


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        logger.info(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")
    if not os.path.exists(app.config['CACHE_FOLDER']):
        os.makedirs(app.config['CACHE_FOLDER'])
        logger.info(f"Created cache folder: {app.config['CACHE_FOLDER']}")

    app.run(debug=True, host='0.0.0.0')
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

from flask import Flask, request, render_template

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
    raise # Re-raise to stop if model isn't loaded

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
    "num py", # Typo from original
    "scikit learn" # Typo from original
}

# --- Text Extraction & Cleaning Functions ---
def clean_resume_text(resume_text):
    """Clean resume text to remove OCR artifacts and normalize formatting."""
    # Remove sequences of numbers (often page numbers, contact info remnants)
    resume_text = re.sub(r'\b(?:\d[\s.-]*){7,}\d\b', ' ', resume_text) # Phone numbers, long sequences of digits
    resume_text = re.sub(r'\b\d+\s+\d+\b', ' ', resume_text) # Common OCR errors with numbers
    # Remove non-ASCII characters, except common punctuation and symbols
    resume_text = re.sub(r'[^\x00-\x7F\u2000-\u206F\u20A0-\u20CF\u2100-\u214F\u2190-\u21FF\u2200-\u22FF\u2500-\u257F\u25A0-\u25FF\u2600-\u26FF\u2700-\u27BF\u2B00-\u2BFF\u2C60-\u2C7F\u2D00-\u2D2F\u2D30-\u2D6F\u2D80-\u2DDF\u2E00-\u2E7F\u2E80-\u2EFF\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u3100-\u312F\u3130-\u318F\u3190-\u31AF\u31C0-\u31EF\u3200-\u32FF\u3300-\u33FF\uFE30-\uFE4F\uFF00-\uFFEF\x20-\x7E]+', ' ', resume_text)
    # Remove emails
    resume_text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', resume_text)
    # Remove phone numbers
    resume_text = re.sub(r'\b(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b', ' ', resume_text)
    # Remove URLs
    resume_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', resume_text, flags=re.IGNORECASE)
    # Replace common bullet points with space or normalize
    resume_text = re.sub(r'[•●■▪\u2022\u2023\u25CF\u25AA\u2043]', ' ', resume_text)
    # Replace multiple spaces/newlines with a single space
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
        logger.debug(f"Extracted text from PDF {file_path}. Length: {len(text)}. First 200 chars: {text[:200]}...")
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF {file_path}: {str(e)}")
        return ""

def extract_text_from_docx(file_path):
    try:
        text = docx2txt.process(file_path)
        logger.debug(f"Extracted text from DOCX {file_path}. Length: {len(text)}. First 200 chars: {text[:200]}...")
        return text
    except Exception as e:
        logger.error(f"Error extracting DOCX {file_path}: {str(e)}")
        return ""

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            logger.debug(f"Extracted text from TXT {file_path}. Length: {len(text)}. First 200 chars: {text[:200]}...")
            return text
    except Exception as e:
        logger.error(f"Error extracting TXT {file_path}: {str(e)}")
        return ""

def extract_text(file_path):
    logger.debug(f"Attempting to extract text from: {file_path}")
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        logger.error(f"Unsupported file format for text extraction: {file_path}")
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

def cache_parsed_resume(parsed_resume, cache_key):
    """Cache parsed resume to disk."""
    cache_path = os.path.join(app.config['CACHE_FOLDER'], f"parsed_resume_{cache_key}.pkl")
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(parsed_resume, f)
        logger.debug(f"Cached parsed resume: {cache_key}")
    except Exception as e:
        logger.error(f"Error caching parsed resume {cache_key}: {str(e)}")

def load_cached_parsed_resume(cache_key):
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
            "quality_feedback": {} # Ensure this is always a dict
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
        # Clean potential markdown code block wrappers
        response_text = re.sub(r'^```json\n|\n```$', '', response_text, flags=re.MULTILINE)
        logger.debug(f"Gemini API raw response: {response_text[:500]}...")
        return response_text
    except Exception as e:
        logger.error(f"Error in Gemini API call: {str(e)}")
        return json.dumps({
            "percentage_match": 0.0,
            "missing_keywords": [{"keyword": f"Gemini API call failed: {str(e)}", "type": "system", "importance": "critical"}],
            "recommendations": ["Check Gemini API key, network, and prompt structure."],
            "quality_feedback": {} # Ensure this is always a dict
        })

# --- Resume Parsing Function ---
def parse_resume(resume_text):
    logger.debug(f"--- Starting parse_resume ---")
    logger.debug(f"Initial resume text received: {resume_text[:200]}...")

    resume_text = clean_resume_text(resume_text)
    logger.debug(f"Cleaned resume text: {resume_text[:200]}...")
    doc = nlp(resume_text)

    # Initialize matchers
    matcher = Matcher(nlp.vocab)
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    # Define section patterns (more robust if they're common headings)
    section_patterns = [
        [{"LOWER": {"IN": ["summary", "about me", "profile"]}}],
        [{"LOWER": {"IN": ["skills", "technical skills", "core competencies", "extracurricular skills", "technologies", "tech skills"]}}],
        [{"LOWER": {"IN": ["experience", "work experience", "professional experience"]}}],
        [{"LOWER": "education"}],
        [{"LOWER": "projects"}],
        [{"LOWER": {"IN": ["achievements", "certifications", "additional information", "awards", "languages", "publications", "hobbies", "interests"]}}]
    ]
    matcher.add("SECTION_HEADING", section_patterns) # Renamed ID to avoid collision with phrase matcher

    # Add soft skills to phrase matcher
    soft_skill_phrases = [nlp.make_doc(phrase) for phrase in SOFT_SKILLS]
    phrase_matcher.add("SOFT_SKILLS_GLOBAL", soft_skill_phrases)

    # Add technical skills to phrase matcher
    all_technical_phrases = [nlp.make_doc(s) for s in TECHNICAL_SKILLS]
    phrase_matcher.add("TECHNICAL_SKILLS_GLOBAL", all_technical_phrases)


    sections_raw_lines = {
        "summary": [], "skills": [], "experience": [],
        "education": [], "projects": [], "additional_info": []
    }
    current_section = None

    lines = [line.strip() for line in resume_text.split('\n') if line.strip()]

    # Iterate through lines to identify sections
    for line_idx, line in enumerate(lines):
        line_doc = nlp(line)
        line_matches = matcher(line_doc) # Use section matcher

        found_section_in_line = False
        for match_id, start, end in line_matches:
            # Check for specific section types based on matched text
            section_text_lower = line_doc[start:end].text.lower()
            
            if any(s in section_text_lower for s in ["summary", "about me", "profile"]):
                current_section = "summary"
            elif any(s in section_text_lower for s in ["skills", "technical skills", "core competencies", "extracurricular skills", "technologies", "tech skills"]):
                current_section = "skills"
            elif "experience" in section_text_lower:
                current_section = "experience"
            elif "education" in section_text_lower:
                current_section = "education"
            elif "projects" in section_text_lower:
                current_section = "projects"
            elif any(s in section_text_lower for s in ["achievements", "certifications", "additional information", "awards", "languages", "publications", "hobbies", "interests"]):
                current_section = "additional_info"
            else:
                continue # If it's a match but not one of our main sections, ignore for section parsing
            found_section_in_line = True
            break # Found a section, stop processing this line for section headers

        if not found_section_in_line:
            # If no section header found, append line to current_section or summary
            if current_section:
                sections_raw_lines[current_section].append(line)
            else:
                # If no section detected yet, assume it's part of the summary/intro
                sections_raw_lines["summary"].append(line)


    # Extract skills globally and specifically from skills section
    extracted_technical_skills = set()
    extracted_soft_skills = set()

    # Extract skills from the dedicated 'skills' section first
    if sections_raw_lines["skills"]:
        skills_text_from_section = " ".join(sections_raw_lines["skills"]).lower()
        skills_doc_section = nlp(skills_text_from_section)
        
        # Use the global phrase matchers for skills within the skills section
        for match_id, start, end in phrase_matcher(skills_doc_section):
            matched_phrase = skills_doc_section[start:end].text.lower()
            if nlp.vocab.strings[match_id] == "TECHNICAL_SKILLS_GLOBAL":
                extracted_technical_skills.add(matched_phrase)
            elif nlp.vocab.strings[match_id] == "SOFT_SKILLS_GLOBAL":
                extracted_soft_skills.add(matched_phrase)
        
        # Also check single words that might be skills (e.g., 'python' directly)
        for token in skills_doc_section:
            token_lower = token.text.lower()
            if token_lower in TECHNICAL_SKILLS:
                extracted_technical_skills.add(token_lower)
            if token_lower in SOFT_SKILLS:
                extracted_soft_skills.add(token_lower)


    # Extract skills from the entire resume text if not already found in dedicated section
    full_resume_doc = nlp(resume_text)
    for match_id, start, end in phrase_matcher(full_resume_doc):
        matched_phrase = full_resume_doc[start:end].text.lower()
        if nlp.vocab.strings[match_id] == "TECHNICAL_SKILLS_GLOBAL":
            extracted_technical_skills.add(matched_phrase)
        elif nlp.vocab.strings[match_id] == "SOFT_SKILLS_GLOBAL":
            extracted_soft_skills.add(matched_phrase)

    # Also check single word tokens in the whole document
    for token in full_resume_doc:
        token_lower = token.text.lower()
        if token_lower in TECHNICAL_SKILLS:
            extracted_technical_skills.add(token_lower)
        if token_lower in SOFT_SKILLS:
            extracted_soft_skills.add(token_lower)


    skills = sorted(list(extracted_technical_skills))
    soft_skills = sorted(list(extracted_soft_skills))

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
    logger.debug(f"Parsed resume FINAL sections: {parsed_data['sections'].keys()}")
    return parsed_data

# --- Resume Quality Analysis Function ---
def analyze_resume_quality(resume_text, sections_content_dict):
    """
    Analyzes resume quality based on rule-based checks.
    Provides default messages for all expected categories.
    """
    quality_feedback = {
        "grammar_issues": ["Grammar and spelling analysis is primarily provided by the AI (Gemini) for advanced contextual review. Check that section for specific insights."],
        "structural_suggestions": [], # Will be populated by rule-based logic
        "conciseness_suggestions": [], # Will be populated by rule-based logic
        "quantifiable_achievements": ["Look for opportunities to add numbers, percentages, or metrics to demonstrate the impact of your work (e.g., 'Increased efficiency by 20%')."],
        "action_verbs": ["Ensure your bullet points start with strong action verbs (e.g., 'Developed', 'Managed', 'Implemented', 'Achieved') to convey impact and ownership."],
    }

    # 1. Structure Analysis (Rule-based)
    detected_sections = list(sections_content_dict.keys())
    standard_sections = ["summary", "experience", "education", "skills", "projects"]
    
    # Check for missing standard sections based on presence of text content
    missing_standard_content = [s for s in standard_sections if not sections_content_dict.get(s, "").strip()]
    if missing_standard_content:
        quality_feedback["structural_suggestions"].append(
            f"Consider ensuring your resume clearly includes standard sections with content, such as: {', '.join([s.replace('_', ' ').title() for s in missing_standard_content])}."
        )

    # Check for consistent bullet points in experience and projects
    for section_name in ["experience", "projects"]:
        if sections_content_dict.get(section_name):
            section_content = sections_content_dict[section_name]
            lines = [line.strip() for line in section_content.split('\n') if line.strip()]
            bullet_start_count = 0
            for line in lines:
                # Check for common bullet characters or start of line dashes/asterisks
                if re.match(r'^\s*[\u2022\u2023\u25CF\u25AA\u2043*-]', line):
                    bullet_start_count += 1
            
            # If a significant portion of lines are not bulleted in a content-heavy section
            if len(lines) > 5 and bullet_start_count < len(lines) * 0.7:
                quality_feedback["structural_suggestions"].append(
                    f"In your '{section_name.replace('_', ' ').title()}' section, consider using more consistent bullet points or a structured format for improved readability and ATS parsing."
                )

    # If no specific structural issues found, add a general positive message
    if not quality_feedback["structural_suggestions"]:
        quality_feedback["structural_suggestions"].append("Basic resume structure appears well-organized for ATS compatibility.")

    # 2. Conciseness Check
    word_count = len(resume_text.split())
    # Estimate pages based on common word count per page (e.g., 500 words/page)
    # Corrected 'n' to 'len(lines)' if 'n' was intended as number of lines in resume_text
    # or ensure 'n' is passed if it means total resumes processed.
    # Assuming 'n' here refers to the total number of resumes uploaded in the batch for this particular context.
    # If it's about pages in *this* resume, it should be derived from word_count / words_per_page
    
    # For a single resume, we check its own word count relative to ideal lengths
    # You were using `n` which is the total count of resumes, not pages of *this* resume.
    # Let's derive `num_pages` directly from `word_count`.
    num_pages = (word_count + 499) // 500 # Integer division, effectively ceil(word_count / 500)

    if num_pages > 2: # More than 2 pages for any resume is usually too long
        quality_feedback["conciseness_suggestions"].append(
            f"Your resume has approximately {num_pages} pages ({word_count} words). For many roles and experience levels, a concise 1-2 page resume is preferred. Consider condensing information."
        )
    elif num_pages > 1 and word_count > 500: # Over 1 page for less experienced candidates
        quality_feedback["conciseness_suggestions"].append(
            f"Your resume has approximately {num_pages} page ({word_count} words). For early career professionals, a single-page resume is often ideal. Review for conciseness."
        )
    
    # If no specific conciseness issues found, add a general positive message
    if not quality_feedback["conciseness_suggestions"]:
        quality_feedback["conciseness_suggestions"].append("Resume length appears appropriate for your content.")

    return quality_feedback

# --- Main Resume Processing Logic ---
def process_resume(resume_file, job_description, input_prompt):
    """Process a single resume file."""
    if not resume_file or not resume_file.filename:
        logger.warning("Empty or no resume file received in process_resume.")
        return None, 0.0, {"percentage_match": 0.0, "missing_keywords": [], "recommendations": [], "quality_feedback": {}}, {"skills": [], "soft_skills": [], "raw_text": "", "sections": {}}, ""

    filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        resume_file.save(filename)
        logger.debug(f"Saved resume file: {filename}")

        resume_text = extract_text(filename)
        if not resume_text.strip():
            logger.error(f"Failed to extract text from {filename} (empty content after extraction).")
            return None, 0.0, {"percentage_match": 0.0, "missing_keywords": [], "recommendations": [], "quality_feedback": {}}, {"skills": [], "soft_skills": [], "raw_text": "", "sections": {}}, ""

        parsed_resume_cache_key = get_resume_cache_key(resume_text)
        parsed_resume = load_cached_parsed_resume(parsed_resume_cache_key)

        if parsed_resume:
            logger.debug(f"Loaded parsed resume from cache for {resume_file.filename}.")
            # Ensure 'sections' key exists and is a dict, re-parse if not
            if "sections" not in parsed_resume or not isinstance(parsed_resume["sections"], dict):
                logger.warning(f"Cached resume for {resume_file.filename} missing 'sections' key or it's malformed. Re-parsing.")
                parsed_resume = parse_resume(resume_text)
                cache_parsed_resume(parsed_resume, parsed_resume_cache_key)
        else:
            parsed_resume = parse_resume(resume_text)
            cache_parsed_resume(parsed_resume, parsed_resume_cache_key)
            logger.debug(f"Parsed and cached resume for {resume_file.filename}.")

        # Perform local quality analysis
        # This will provide default messages for all quality categories
        # and populate structural/conciseness based on rules.
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
            "quality_feedback": {} # Will be populated by merging local and Gemini feedback
        }

        try:
            response_json = json.loads(gemini_response_raw)
            percentage = float(response_json.get("percentage_match", 0.0))
            recommendation_data["percentage_match"] = percentage
            recommendation_data["missing_keywords"] = response_json.get("missing_keywords", [])
            recommendation_data["recommendations"] = response_json.get("recommendations", [])
            
            # --- MERGING QUALITY FEEDBACK ---
            gemini_quality_feedback = response_json.get("quality_feedback", {})
            
            final_quality_feedback = {}
            # Start with local feedback (defaults and rule-based insights)
            final_quality_feedback.update(local_quality_feedback) 

            # Overlay Gemini's feedback intelligently:
            for key, value in gemini_quality_feedback.items():
                if isinstance(value, list) and value: # If Gemini provided a non-empty list for a category
                    final_quality_feedback[key] = value
                elif key in final_quality_feedback and isinstance(final_quality_feedback[key], list) and not final_quality_feedback[key]:
                    # If local had an empty list, and Gemini also has an empty list, keep Gemini's if preferred,
                    # or just keep the default empty one. For robustness, if Gemini explicitly sent an empty list,
                    # it means it found nothing, so we could overwrite our default positive message with an empty list.
                    # For this case, let's prioritize Gemini's empty list if it explicitly gave one.
                    final_quality_feedback[key] = value # Take Gemini's (might be empty list)
                elif key not in final_quality_feedback: # If Gemini provided a key not initially in local_quality_feedback
                    final_quality_feedback[key] = value # Add it as is (might be empty list)
            
            recommendation_data["quality_feedback"] = final_quality_feedback
            # --- END MERGING QUALITY FEEDBACK ---

            logger.debug("Successfully parsed Gemini JSON response and merged quality feedback.")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for Gemini response: {e}. Raw response: {gemini_response_raw[:500]}...")
            # Fallback logic for malformed JSON
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
                    # Attempt to parse as list of dicts first
                    temp_list = json.loads(f"[{keywords_str}]")
                    if all(isinstance(item, dict) and "keyword" in item for item in temp_list):
                        recommendation_data["missing_keywords"] = temp_list
                    else:
                        raise ValueError("Not a list of keyword dicts")
                except (json.JSONDecodeError, ValueError):
                    # Fallback to simple string extraction if dict parsing fails
                    if keywords_str:
                        missing_keywords_list = [kw.strip().strip('"') for kw in re.split(r',\s*(?="|\b)', keywords_str) if kw.strip() and not kw.strip().startswith('{') and not kw.strip().endswith('}')]
                        # Try to extract dicts if regex finds them
                        keyword_objects = re.findall(r'{"keyword":"(.*?)"(?:,"type":"(.*?)"(?:,"importance":"(.*?)"|)|)}', keywords_str)
                        if keyword_objects:
                            for kw, type_val, imp_val in keyword_objects:
                                item = {"keyword": kw}
                                if type_val: item["type"] = type_val
                                if imp_val: item["importance"] = imp_val
                                if item not in recommendation_data["missing_keywords"]: # Avoid duplicates
                                    recommendation_data["missing_keywords"].append(item)
                        elif missing_keywords_list: # Fallback to just list of strings if no dicts
                            recommendation_data["missing_keywords"].extend([
                                {"keyword": k, "type": "unknown", "importance": "unknown"} for k in missing_keywords_list
                                if {"keyword": k, "type": "unknown", "importance": "unknown"} not in recommendation_data["missing_keywords"] # Avoid duplicates
                            ])
                except Exception as ex:
                    logger.debug(f"Failed to extract keywords from malformed JSON string '{keywords_str}' using fallback: {ex}")
                    pass # Continue processing even if keyword parsing fails

            # Fallback for quality_feedback from malformed JSON
            quality_feedback_match = re.search(r'"quality_feedback"\s*:\s*({.*?})', gemini_response_raw, re.DOTALL)
            if quality_feedback_match:
                try:
                    quality_feedback_str = quality_feedback_match.group(1)
                    parsed_gemini_quality = json.loads(quality_feedback_str)
                    # Merge even on parsing error for best effort
                    for key, value in parsed_gemini_quality.items():
                        if isinstance(value, list) and value:
                            recommendation_data["quality_feedback"][key] = value
                        elif key not in recommendation_data["quality_feedback"]: # Only add if not already present from local
                            recommendation_data["quality_feedback"][key] = value
                except json.JSONDecodeError as ex:
                    logger.debug(f"Failed to extract quality_feedback from malformed JSON: {ex}")
                    # Add a general error message to quality feedback if parsing fails
                    recommendation_data["quality_feedback"]["parsing_error"] = [f"Failed to parse quality_feedback from AI: {str(ex)}. Check raw response."]
            
            # Ensure all categories from local_quality_feedback are present in final_quality_feedback
            # if they weren't explicitly provided by Gemini or parsed from malformed JSON.
            # This ensures defaults are kept if Gemini provides nothing for a category.
            for key, value in local_quality_feedback.items():
                # If the key is not in recommendation_data["quality_feedback"] OR
                # if it exists but is an empty list/falsey, then use the local value.
                # This ensures local defaults/rule-based feedback is shown unless Gemini provides something specific.
                if key not in recommendation_data["quality_feedback"] or \
                   (isinstance(recommendation_data["quality_feedback"].get(key), list) and not recommendation_data["quality_feedback"].get(key)):
                    recommendation_data["quality_feedback"][key] = value

            recommendation_data["recommendations"].append(f"Full Raw Gemini Response (for debugging): {gemini_response_raw}")

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
            pass # Ignore errors during cleanup
        error_recommendation_data = {
            "percentage_match": 0.0,
            "missing_keywords": [],
            "recommendations": [f"Critical error processing {resume_file.filename}: {str(e)}", "Please check server logs for details."],
            "quality_feedback": {"system_error": [f"Critical processing error: {str(e)}"]} # Ensure this is a list
        }
        return None, 0.0, error_recommendation_data, {"skills": [], "soft_skills": [], "raw_text": "", "sections": {}}, ""

# --- Flask Routes ---
@app.route("/")
def index(): # Renamed to index as it's the root
    return render_template('main.html')

@app.route("/matchresume")
def matchresume(): # Now this is the form page
    # Pass job_description if it was sent back from a failed form submission
    job_description_text = request.args.get('job_description', '')
    return render_template('home.html', job_description=job_description_text)


@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        # Add a breakpoint here if running in an IDE, or use logging extensively
        job_description = request.form.get('job_description', '').strip()
        resume_files = request.files.getlist('resumes')

        logger.info(f"--- Form Submission Received ---")
        logger.info(f"Number of resume files received: {len(resume_files)}")
        logger.info(f"Job description length: {len(job_description)} chars (empty if 0)")
        logger.debug(f"Job description content (first 200 chars): '{job_description[:200]}'")
        
        if not job_description:
            logger.warning("Job description is empty.")
        if not resume_files:
            logger.warning("No resume files were uploaded.")

        # --- Initial Validation ---
        if not resume_files or not job_description:
            logger.warning("Returning to home.html due to missing job description or resumes.")
            # Pass job_description back so user doesn't lose their input
            return render_template('home.html', message="Please upload resumes and enter a job description.", job_description=job_description)

        processed_results = []

        # Current year for footer (can be dynamic, but static is fine for template example)
        current_year = 2025 

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

        **Instructions for Resume Quality Feedback (beyond matching):**
        - **Grammar & Spelling:** Identify and list specific instances of grammatical errors, typos, punctuation mistakes, and awkward phrasing. If no significant errors, state that or mention the resume appears well-written. Focus on high-impact errors. If no issues, state "No significant grammatical errors or typos detected. The resume is well-written."
        - **Resume Structure & Formatting:** Assess if the resume follows standard, ATS-friendly formatting (e.g., clear, consistent headings like 'Work Experience', 'Education', 'Skills', 'Projects'; proper use of bullet points; avoidance of excessive graphics). Provide concrete suggestions for structural improvements. If no issues, state "Resume structure and formatting appear clean and ATS-friendly."
        - **Quantifiable Achievements:** Identify opportunities where the candidate could add quantifiable results or metrics (numbers, percentages, currency, scale) to demonstrate impact more effectively in their experience and project descriptions. Provide examples of how they might rephrase points to include impact. If no issues, state "Your resume effectively highlights quantifiable achievements."
        - **Action Verbs:** Note if the resume predominantly uses passive voice or weak verbs. Suggest areas where stronger, more impactful action verbs could be used to start bullet points or describe responsibilities (e.g., "managed," "developed," "implemented," "achieved"). If no issues, state "Strong action verbs are used effectively throughout the resume."
        - **Conciseness:** Evaluate if the resume's length is appropriate for the candidate's experience level (e.g., 1 page for early career, 1-2 pages for mid-career). Suggest areas for condensation if it appears too long, or expansion if too brief. If no issues, state "Resume length appears appropriate and concise."

        Return the entire response in a single JSON format. Ensure all keys and string values are properly quoted.
        ```json
        {{
          "percentage_match": <number between 0 and 100, float, precise to one decimal place, e.g., 82.5>,
          "missing_keywords": [
            {{"keyword": "<string>", "type": "technical/soft/general", "importance": "critical/important/optional"}},
            ...
          ],
          "recommendations": [<string>, ...],
          "quality_feedback": {{
            "grammar_issues": [<string>, ...],
            "structural_suggestions": [<string>, ...],
            "quantifiable_achievements": [<string>, ...],
            "action_verbs": [<string>, ...],
            "conciseness_suggestions": [<string>, ...]
          }}
        }}
        ```
        """

        # Use ThreadPoolExecutor to process resumes concurrently
        # Max workers can be adjusted based on system resources and API rate limits
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor: 
            # Submit each resume file to the executor for processing
            future_to_resume_file = {
                executor.submit(process_resume, resume_file, job_description, input_prompt): resume_file
                for resume_file in resume_files
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_resume_file):
                original_file = future_to_resume_file[future]
                try:
                    filename, percentage, recommendation_data, parsed_resume, resume_text = future.result()
                    if filename: # Check if processing was successful (filename is not None)
                        processed_results.append({
                            "filename": filename,
                            "score": percentage, # Initial score from Gemini
                            "recommendation_data": recommendation_data,
                            "parsed_resume": parsed_resume,
                            "resume_text": resume_text
                        })
                    else:
                        logger.error(f"Processing of {original_file.filename} failed. Error data: {recommendation_data}")
                        # Append a "failed" entry to show in results if needed, or skip
                        processed_results.append({
                            "filename": original_file.filename or "Unnamed_Resume_Failed",
                            "score": 0.0, # Indicate failure with 0 score
                            "recommendation_data": recommendation_data, # Contains error messages
                            "parsed_resume": {"skills": [], "soft_skills": [], "raw_text": "", "sections": {}},
                            "resume_text": ""
                        })
                except Exception as e:
                    logger.exception(f"Unhandled exception during concurrent processing of {original_file.filename}:")
                    # Append a "critically failed" entry
                    processed_results.append({
                        "filename": original_file.filename or "Unnamed_Resume_Critically_Failed",
                        "score": 0.0,
                        "recommendation_data": {
                            "percentage_match": 0.0,
                            "missing_keywords": [{"keyword": "System Error", "type": "system", "importance": "critical"}],
                            "recommendations": [f"Critical processing error for {original_file.filename}: {str(e)}", "Please check server logs."],
                            "quality_feedback": {"system_error": [f"Critical system error: {str(e)}. Consult server logs."]}
                        },
                        "parsed_resume": {"skills": [], "soft_skills": [], "raw_text": "", "sections": {}},
                        "resume_text": ""
                    })

        # Final check after all processing attempts
        if not processed_results or all(res["score"] == 0.0 and not res["resume_text"].strip() for res in processed_results):
            logger.warning("No valid resumes processed successfully after all attempts. Returning to home.html.")
            return render_template('home.html', message="No valid resumes could be processed. Ensure files are readable and contain text.", job_description=job_description)

        # --- TF-IDF Calculation ---
        all_texts_for_tfidf_input = []
        original_indices_map = [] 

        job_desc_tfidf_idx = -1
        if job_description.strip():
            all_texts_for_tfidf_input.append(job_description)
            job_desc_tfidf_idx = 0

        for i, res in enumerate(processed_results):
            if res["resume_text"].strip(): # Only include valid extracted resume texts for TF-IDF
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

                        for i_sim, score in enumerate(similarities):
                            original_res_idx = original_indices_map[i_sim]
                            tfidf_scores[original_res_idx] = round(score * 100, 2)
                    else:
                        logger.warning("No valid resume vectors for TF-IDF similarity calculation after filtering.")
                else:
                    logger.warning("TF-IDF vectorization resulted in empty features or job description not available for vectorization.")
            else:
                logger.warning("Insufficient valid texts for TF-IDF vectorization (need job description and at least one resume). Skipping TF-IDF.")
        except Exception as e:
            logger.error(f"Error during TF-IDF vectorization: {str(e)}")


        final_resumes_for_display = []
        GEMINI_WEIGHT = 0.7
        TFIDF_WEIGHT = 0.3

        for i, res in enumerate(processed_results):
            gemini_score = res["score"] # This is the percentage from Gemini
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
                "score": round(combined_score, 2), # This is the final combined score
                "skills": res["parsed_resume"]["skills"],
                "soft_skills": res["parsed_resume"]["soft_skills"],
                "recommendation_data": res["recommendation_data"] 
            })

        top_resumes = sorted(final_resumes_for_display, key=lambda x: x['score'], reverse=True)

        logger.info(f"Top matching resumes calculated: {[r['filename'] for r in top_resumes]}")
        
        # --- Final Render to ats.html ---
        return render_template(
            'ats.html',
            job_description=job_description,
            top_resumes=top_resumes,
            current_year=current_year 
        )

    # This part should ideally not be reached for POST requests, but is a fallback for GET requests to /matcher
    return render_template('home.html', message="Invalid request method. Please use the form to submit.")


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        logger.info(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")
    if not os.path.exists(app.config['CACHE_FOLDER']):
        os.makedirs(app.config['CACHE_FOLDER'])
        logger.info(f"Created cache folder: {app.config['CACHE_FOLDER']}")

    app.run(debug=True, host='0.0.0.0', port=5000)
from flask import Flask, request, render_template 
import os
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import Matcher, PhraseMatcher # PhraseMatcher is here
import re
import logging
import google.generativeai as genai
import json
import concurrent.futures
import pickle
import hashlib
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

current_year = datetime.now().year

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads/'
app.config['CACHE_FOLDER'] = 'Cache/'


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set. Please set it before running the application.")

else:
    genai.configure(api_key=GEMINI_API_KEY)
    print(f"DEBUG: GEMINI_API_KEY loaded into app: '{GEMINI_API_KEY}'") 

try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {str(e)}. Please run 'python -m spacy download en_core_web_sm'")
    raise 
SOFT_SKILLS = {
    "communication", "teamwork", "leadership", "problem-solving", "adaptability",
    "time management", "collaboration", "creativity", "work ethic", "interpersonal skills",
    "organizational skills", "critical thinking", "negotiation", "conflict resolution",
    "emotional intelligence", "decision making", "presentation skills", "active listening",
    "attention to detail", "analytical skills"
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
    "aws glue",   # From AI-FRAUD DETECTION SYSTEM
    "aws iam",    # From AI-FRAUD DETECTION SYSTEM
    "next.js",    # From WORK EXPERIENCE
    "react.js",   # From WORK EXPERIENCE
    "generative ai", # From WORK EXPERIENCE
    "tkinter",    # From LEARNING MANAGEMENT SYSTEM
    "socket.io",  # From REAL-TIME CHAT APPLICATION
    "num py",     # Corrected "NumPy" potentially
    "scikit learn", # Corrected "Scikit-learn" potentially
    "power bi" # Ensure consistent casing
}

def clean_resume_text(resume_text):
    """Clean resume text to remove OCR artifacts and normalize formatting."""
    # Remove repetitive numbers (e.g., "4 4 19 19...")
    resume_text = re.sub(r'\b(\d+\s+)+\d+\b', ' ', resume_text)
    # Remove non-ASCII characters and special symbols
    resume_text = re.sub(r'[^\x00-\x7F]+', ' ', resume_text)
    # Remove email addresses, phone numbers, and social media links (more general URL regex)
    resume_text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', resume_text) # Email
    resume_text = re.sub(r'\b\+?\d{10,}\b', ' ', resume_text) # Phone numbers
    # More robust URL detection (adapted to be more general)
    resume_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', resume_text, flags=re.IGNORECASE)
    # Replace common bullet characters with space or nothing for cleaner splitting later
    resume_text = re.sub(r'[•●■▪]', ' ', resume_text) # Handles different bullet types
    # Normalize whitespace
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
        logger.debug(f"Extracted text from PDF {file_path}: {text[:200]}...") # Log more text
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF {file_path}: {str(e)}")
        return ""

def extract_text_from_docx(file_path):
    try:
        text = docx2txt.process(file_path)
        logger.debug(f"Extracted text from DOCX {file_path}: {text[:200]}...") # Log more text
        return text
    except Exception as e:
        logger.error(f"Error extracting DOCX {file_path}: {str(e)}")
        return ""

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            logger.debug(f"Extracted text from TXT {file_path}: {text[:200]}...") # Log more text
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

# Generate cache key for Gemini response based on both JD and resume
def get_gemini_response_cache_key(job_description, resume_text, prompt):
    combined_input = job_description + "|||" + resume_text + "|||" + prompt
    return hashlib.md5(combined_input.encode('utf-8')).hexdigest()

# Cache Gemini response
def cache_gemini_response(gemini_response, cache_key):
    cache_path = os.path.join(app.config['CACHE_FOLDER'], f"gemini_{cache_key}.pkl")
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(gemini_response, f)
        logger.debug(f"Cached Gemini response: {cache_key}")
    except Exception as e:
        logger.error(f"Error caching Gemini response {cache_key}: {str(e)}")

# Load cached Gemini response
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

def get_gemini_response(job_description, resume_text, prompt):
    if not GEMINI_API_KEY:
        logger.warning("Gemini API key not set, skipping API call and returning default error.")
        return json.dumps({
            "percentage_match": 0.0,
            "missing_keywords": [{"keyword": "Gemini API key not configured.", "type": "system", "importance": "critical"}],
            "recommendations": ["Set GEMINI_API_KEY environment variable."]
        })

    # Add generation_config to ensure deterministic output
    generation_config = {
        "temperature": 0.0, # CRITICAL FOR CONSISTENCY
        "top_p": 1.0,
        "top_k": 1,
    }

    try:
        model = genai.GenerativeModel('gemini-1.5-flash') # Using flash as specified in original code

        input_text = f"Job Description:\n{job_description}\n\nResume:\n{resume_text}"

        # Use generation_config for more deterministic output
        response = model.generate_content([input_text, prompt], generation_config=generation_config)

        response_text = response.text
        # Clean markdown code fences from Gemini response
        response_text = re.sub(r'^```json\n|\n```$', '', response_text, flags=re.MULTILINE)
        logger.debug(f"Gemini API raw response: {response_text[:500]}...") # Log more of the response for debugging
        return response_text
    except Exception as e:
        logger.error(f"Error in Gemini API call: {str(e)}")
        return json.dumps({
            "percentage_match": 0.0,
            "missing_keywords": [{"keyword": f"Gemini API call failed: {str(e)}", "type": "system", "importance": "critical"}],
            "recommendations": ["Check Gemini API key, network, and prompt structure."]
        })


def parse_resume(resume_text):
    logger.debug(f"--- Starting parse_resume ---")
    logger.debug(f"Initial resume text received: {resume_text[:200]}...")

    # Clean resume text
    resume_text = clean_resume_text(resume_text)
    logger.debug(f"Cleaned resume text: {resume_text[:200]}...")
    doc = nlp(resume_text) # Process the whole cleaned resume for broader matching

    # Initialize Matcher for section detection
    matcher = Matcher(nlp.vocab)
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER") # Re-initialize for fresh use

    # Define patterns for section headings (case-insensitive, flexible)
    section_patterns = [
        [{"LOWER": {"IN": ["summary", "about me", "profile"]}}],
        [{"LOWER": {"IN": ["skills", "technical skills", "core competencies", "extracurricular skills", "technologies", "tech skills"]}}],
        [{"LOWER": {"IN": ["experience", "work experience", "professional experience"]}}],
        [{"LOWER": "education"}],
        [{"LOWER": "projects"}],
        [{"LOWER": {"IN": ["achievements", "certifications", "additional information", "awards", "languages"]}}]
    ]

    # REMOVED overwrite=True from here
    matcher.add("SECTION", section_patterns)

    # Add soft skill phrases to PhraseMatcher once
    soft_skill_phrases = [nlp.make_doc(phrase) for phrase in SOFT_SKILLS]
    # REMOVED overwrite=True from here
    phrase_matcher.add("SOFT_SKILLS", soft_skill_phrases) 

    # Find section headings and associate content
    sections = {
        "summary": [], "skills": [], "experience": [],
        "education": [], "projects": [], "additional_info": []
    }
    current_section = None

    lines = [line.strip() for line in resume_text.split('\n') if line.strip()]
    logger.debug(f"Total {len(lines)} non-empty lines after initial split.")

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
            elif any(s in section_text_lower for s in ["achievements", "certifications", "additional information", "awards", "languages"]):
                current_section = "additional_info"
            else:
                continue # If it's a match but not one of our main sections, ignore for section parsing
            found_section_in_line = True
            logger.debug(f"L{line_idx}: Detected section: '{section_text_lower}' -> assigned to {current_section}")
            break

        if not found_section_in_line and current_section:
            sections[current_section].append(line)
        elif not found_section_in_line and not current_section:
            sections["summary"].append(line) # Default to summary if no section header found yet

    logger.debug(f"Sections identified: { {k: len(v) for k, v in sections.items()} }")
    logger.debug(f"Content of 'skills' section: {' '.join(sections['skills'])[:200]}...")


    # Extract skills (technical and soft)
    skills = []
    soft_skills = []

    # --- TECHNICAL SKILLS EXTRACTION (More Aggressive Strategy) ---

    # 1. First, process the designated 'skills' section if it was found
    if sections["skills"]:
        skills_text_from_section = " ".join(sections["skills"]).lower()
        logger.debug(f"Attempting to extract skills from dedicated 'skills' section: '{skills_text_from_section}'")

        # Create a specific PhraseMatcher for the exact TECHNICAL_SKILLS from the list
        tech_matcher_for_section = PhraseMatcher(nlp.vocab, attr="LOWER")
        tech_phrases_for_section = [nlp.make_doc(s) for s in TECHNICAL_SKILLS]
        # REMOVED overwrite=True from here
        tech_matcher_for_section.add("EXACT_TECH_SKILLS_SECTION", tech_phrases_for_section)

        skills_doc_section = nlp(skills_text_from_section)
        for match_id, start, end in tech_matcher_for_section(skills_doc_section):
            matched_skill = skills_doc_section[start:end].text.lower()
            if matched_skill not in skills:
                skills.append(matched_skill)
                logger.debug(f"Found exact tech skill in dedicated section: '{matched_skill}'")
            
        # Additionally, split by common delimiters to catch single words that might not be phrases
        # (e.g., if "Java" is listed with no space/punctuation after it, sometimes PhraseMatcher needs context)
        words_in_skills_section = re.findall(r'\b[a-zA-Z0-9+#.-]+\b', skills_text_from_section) # More robust regex for words
        for word in words_in_skills_section:
            word_lower = word.strip().lower()
            if word_lower in TECHNICAL_SKILLS and word_lower not in skills: # Avoid adding duplicates already found by phrase matcher
                skills.append(word_lower)
                logger.debug(f"Found single word tech skill in dedicated section: '{word_lower}'")
    else:
        logger.debug("Dedicated 'skills' section not found or is empty.")


    # 2. Second, scan the entire document for any TECHNICAL_SKILLS (less strict, more contextual)
    # This acts as a fallback if sectioning fails or skills are mentioned elsewhere.
    full_resume_doc = nlp(resume_text) # Use the full cleaned document
    logger.debug("Scanning entire document for technical skills (contextual search)...")

    # Add ALL TECHNICAL_SKILLS as phrases for the global search
    all_tech_phrases = [nlp.make_doc(s) for s in TECHNICAL_SKILLS]
    # REMOVED overwrite=True from here
    phrase_matcher.add("ALL_TECHNICAL_SKILLS_GLOBAL", all_tech_phrases)

    for match_id, start, end in phrase_matcher(full_resume_doc):
        if nlp.vocab.strings[match_id] == "ALL_TECHNICAL_SKILLS_GLOBAL":
            matched_skill = full_resume_doc[start:end].text.lower()
            if matched_skill not in skills:
                skills.append(matched_skill)
                logger.debug(f"Found contextual tech skill in document: '{matched_skill}'")

    # Final check for single tokens across the entire document
    for token in full_resume_doc:
        token_text = token.text.lower()
        if token_text in TECHNICAL_SKILLS and token_text not in skills:
            skills.append(token_text)
            logger.debug(f"Found single word tech skill in document (token-level): '{token_text}'")


    # --- SOFT SKILLS EXTRACTION ---
    # Apply PhraseMatcher on the whole cleaned resume text for soft skills
    logger.debug("Scanning entire document for soft skills...")
    for match_id, start, end in phrase_matcher(doc): # Use original 'doc' from cleaned_resume_text
        if nlp.vocab.strings[match_id] == "SOFT_SKILLS":
            matched_soft_skill = doc[start:end].text.lower()
            if matched_soft_skill not in soft_skills:
                soft_skills.append(matched_soft_skill)
                logger.debug(f"Matched soft skill: {matched_soft_skill}")


    # Deduplicate and sort lists for clean output
    skills = sorted(list(set(skills)))
    soft_skills = sorted(list(set(soft_skills)))

    parsed_data = {
        "skills": skills,
        "soft_skills": soft_skills,
        "raw_text": resume_text,
        "sections": {k: " ".join(v) for k, v in sections.items()} # Join section lines into a single string for storage
    }

    logger.debug(f"--- Finished parse_resume ---")
    logger.debug(f"Parsed resume FINAL technical skills: {parsed_data['skills']}")
    logger.debug(f"Parsed resume FINAL soft skills: {parsed_data['soft_skills']}")
    logger.debug(f"Parsed resume FINAL sections: {parsed_data['sections'].keys()}")
    
    return parsed_data

# Rest of your main.py code (get_gemini_response_cache_key, cache_gemini_response, etc.)
# ... (all functions after parse_resume, including the /matcher route and if __name__ == '__main__': block) ...
# (The TF-IDF fix you already have is correctly implemented and was not part of this error)

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

def process_resume(resume_file, job_description, input_prompt):
    """Process a single resume file."""
    if not resume_file or not resume_file.filename:
        logger.warning("Empty or no resume file received in process_resume.")
        return None, 0.0, {"percentage_match": 0.0, "missing_keywords": [{"keyword": "No valid resume file uploaded.", "type": "system", "importance": "critical"}], "recommendations": []}, {"skills": [], "soft_skills": [], "raw_text": ""}, ""

    filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        resume_file.save(filename)
        logger.debug(f"Saved resume file: {filename}")

        resume_text = extract_text(filename)
        if not resume_text.strip():
            logger.error(f"Failed to extract text from {filename} (empty content).")
            return None, 0.0, {"percentage_match": 0.0, "missing_keywords": [{"keyword": "Failed to extract readable text from resume.", "type": "system", "importance": "critical"}], "recommendations": []}, {"skills": [], "soft_skills": [], "raw_text": ""}, ""

        # Check for cached parsed resume
        parsed_resume_cache_key = get_resume_cache_key(resume_text)
        parsed_resume = load_cached_resume(parsed_resume_cache_key)

        if parsed_resume:
            logger.debug(f"Loaded parsed resume from cache for {resume_file.filename}.")
        else:
            parsed_resume = parse_resume(resume_text)
            cache_resume(parsed_resume, parsed_resume_cache_key)
            logger.debug(f"Parsed and cached resume for {resume_file.filename}.")

        # Check for cached Gemini API response
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
        recommendation_data = { # Store structured data
            "percentage_match": 0.0,
            "missing_keywords": [],
            "recommendations": []
        }

        try:
            response_json = json.loads(gemini_response_raw)
            percentage = float(response_json.get("percentage_match", 0.0))
            recommendation_data["percentage_match"] = percentage
            recommendation_data["missing_keywords"] = response_json.get("missing_keywords", [])
            recommendation_data["recommendations"] = response_json.get("recommendations", [])
            logger.debug("Successfully parsed Gemini JSON response.")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for Gemini response: {e}. Raw response: {gemini_response_raw[:500]}...")
            # Fallback regex for percentage if full JSON parsing fails
            match = re.search(r'"percentage_match"\s*:\s*(\d+\.?\d*)', gemini_response_raw)
            if match:
                percentage = float(match.group(1))
                recommendation_data["percentage_match"] = percentage
                recommendation_data["recommendations"].append(f"Warning: Gemini response JSON malformed. Percentage extracted by regex: {percentage}%. Raw response saved below.")
            else:
                percentage = 0.0 # Ensure percentage is 0 if nothing can be extracted
                recommendation_data["recommendations"].append(f"Error: Gemini response JSON malformed, and percentage not found via regex. Raw response saved below.")

            # Attempt to extract missing keywords from raw text if JSON failed
            missing_kw_match = re.search(r'"missing_keywords"\s*:\s*\[(.*?)\]', gemini_response_raw, re.DOTALL) # re.DOTALL to match across lines
            if missing_kw_match:
                try:
                    keywords_str = missing_kw_match.group(1).strip()
                    temp_list = []
                    try: # Try to parse as valid JSON list of dicts first
                        temp_list = json.loads(f"[{keywords_str}]")
                        if all(isinstance(item, dict) and "keyword" in item for item in temp_list):
                            recommendation_data["missing_keywords"] = temp_list
                        else: # If not proper dicts, fall back to simple string parsing
                            raise ValueError("Not a list of keyword dicts")
                    except (json.JSONDecodeError, ValueError):
                        if keywords_str: # Fallback to simple comma-separated string parsing
                            missing_keywords_list = [kw.strip().strip('"') for kw in re.split(r',\s*(?="|\b)', keywords_str) if kw.strip()]
                            recommendation_data["missing_keywords"] = [{"keyword": k, "type": "unknown", "importance": "unknown"} for k in missing_keywords_list]
                except Exception as ex:
                    logger.debug(f"Failed to extract keywords from malformed JSON string '{keywords_str}' using fallback: {ex}")
                    pass # Silently fail if keyword extraction from malformed string also fails

            # Always add the raw (potentially malformed) response for debugging
            recommendation_data["recommendations"].append(f"Full Raw Gemini Response (for debugging): {gemini_response_raw}")


        # Clean up the uploaded file after processing
        try:
            os.remove(filename)
            logger.debug(f"Removed temporary file: {filename}")
        except OSError as e:
            logger.error(f"Error removing file {filename}: {e}")

        # Pass recommendation_data directly, not just a string
        return resume_file.filename, percentage, recommendation_data, parsed_resume, resume_text
    except Exception as e:
        logger.error(f"Critical error processing file {resume_file.filename}: {str(e)}")
        # Ensure file is removed even on unexpected errors
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except OSError:
            pass
        # Return structured error data
        error_recommendation_data = {
            "percentage_match": 0.0,
            "missing_keywords": [{"keyword": "Processing Error", "type": "system", "importance": "critical"}],
            "recommendations": [f"Critical error processing {resume_file.filename}: {str(e)}", "Please check server logs for details."]
        }
        return None, 0.0, error_recommendation_data, {"skills": [], "soft_skills": [], "raw_text": "", "sections": {}}, ""

@app.route("/")
def index():
    return render_template('main.html')

@app.route("/matchresume")
def matchresume(): # Now this is the form page
    # Pass job_description if it was sent back from a failed form submission
    job_description_text = request.args.get('job_description', '')
    return render_template('home.html', job_description=job_description_text)


@app.route('/matcher', methods=['POST'])
def matcher():
    current_year = datetime.now().year
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

        processed_results = [] # Store all results here, then sort

        # Gemini API prompt for percentage match and recommendations
        input_prompt = """
        You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality.
        Evaluate the resume against the provided job description. Your goal is to provide a highly accurate and granular percentage match,
        along with detailed missing keywords and actionable recommendations for improvement.

        Strictly adhere to the following percentage calculation weighting:
        - Technical Skills (e.g., programming languages, frameworks, tools, algorithms): 50%
        - Work Experience & Projects (relevance, depth, impact, quantifiable achievements, duration): 30%
        - Soft Skills (e.g., communication, problem-solving, teamwork, leadership): 10%
        - Education & Certifications (relevance of degree, institutions, relevant courses): 10%

        For each missing keyword, specify if it's a critical technical skill, a key soft skill, or a general requirement.
        Provide recommendations that are specific, actionable, and constructive (e.g., "Quantify achievements in 'X' experience entry," "Add more details on your 'Python' projects," "Elaborate on your 'leadership' experience with specific examples").

        Return the response in JSON format. Ensure all keys and string values are properly quoted.
        ```json
        {{
          "percentage_match": <number between 0 and 100, float, precise to one decimal place, e.g., 82.5>,
          "missing_keywords": [
            {{"keyword": "<string>", "type": "technical/soft/general", "importance": "critical/important/optional"}},
            ...
          ],
          "recommendations": [<string>, ...]
        }
        ```
        """

        # Process resumes in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
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

        # Prepare texts for TF-IDF vectorization
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

                    if (job_desc_tfidf_idx + 1) < len(all_texts_for_tfidf_input):
                        resume_vectors_for_tfidf = vectors[job_desc_tfidf_idx + 1:]

                        if resume_vectors_for_tfidf.shape[0] > 0:
                            similarities = cosine_similarity(job_vector, resume_vectors_for_tfidf).flatten()

                            for i, score in enumerate(similarities):
                                original_res_idx = original_indices_map[i]
                                tfidf_scores[original_res_idx] = round(score * 100, 2)
                        else:
                            logger.warning("No valid resume vectors for TF-IDF similarity calculation after filtering.")
                    else:
                        logger.warning("No resume texts available for TF-IDF similarity comparison with job description.")
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
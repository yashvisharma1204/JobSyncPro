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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads/'
app.config['CACHE_FOLDER'] = 'Cache/' # This will now also store Gemini responses

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini API
# !!! IMPORTANT: NEVER HARDCODE API KEYS IN PRODUCTION CODE. USE ENVIRONMENT VARIABLES.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set. Please set it before running the application.")
    # For production, you might raise an exception or exit here.
    # For dev, we'll continue but Gemini calls will return errors.
else:
    genai.configure(api_key=GEMINI_API_KEY)


# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {str(e)}. Please run 'python -m spacy download en_core_web_sm'")
    raise # Re-raise to prevent the app from running without a critical dependency

# Expanded list of soft skills
SOFT_SKILLS = {
    "communication", "teamwork", "leadership", "problem-solving", "adaptability",
    "time management", "collaboration", "creativity", "work ethic", "interpersonal skills",
    "organizational skills", "critical thinking", "negotiation", "conflict resolution",
    "emotional intelligence", "decision making", "presentation skills", "active listening",
    "attention to detail", "analytical skills"
}

# Expanded list of technical skills for validation (ensuring multi-word skills are present)
TECHNICAL_SKILLS = {
    "python", "java", "javascript", "html", "css", "mysql", "mongodb", "pandas",
    "numpy", "matplotlib", "scikit-learn", "power bi", "aws", "azure", "git",
    "github", "flask", "socketio", "pyspark", "databricks", "gemini api", "seaborn",
    "apache spark", "xgboost", "linear regression", "decision tree", "rest apis",
    "html/css", "ci/cd", "angular js", "react", "oracle", "machine learning",
    "deep learning", "natural language processing", "data analysis", "data visualization",
    "cloud computing", "devops", "kubernetes", "docker", "sql", "excel", "tableau",
    "r", "c++", "c#", "scala", "go", "spring boot", "node.js", "typescript"
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
        logger.debug(f"Extracted text from PDF {file_path}: {text[:100]}...")
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF {file_path}: {str(e)}")
        return ""

def extract_text_from_docx(file_path):
    try:
        text = docx2txt.process(file_path)
        logger.debug(f"Extracted text from DOCX {file_path}: {text[:100]}...")
        return text
    except Exception as e:
        logger.error(f"Error extracting DOCX {file_path}: {str(e)}")
        return ""

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            logger.debug(f"Extracted text from TXT {file_path}: {text[:100]}...")
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

# NEW FUNCTION: Generate cache key for Gemini response based on both JD and resume
def get_gemini_response_cache_key(job_description, resume_text, prompt):
    combined_input = job_description + "|||" + resume_text + "|||" + prompt
    return hashlib.md5(combined_input.encode('utf-8')).hexdigest()

# NEW FUNCTION: Cache Gemini response
def cache_gemini_response(gemini_response, cache_key):
    cache_path = os.path.join(app.config['CACHE_FOLDER'], f"gemini_{cache_key}.pkl")
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(gemini_response, f)
        logger.debug(f"Cached Gemini response: {cache_key}")
    except Exception as e:
        logger.error(f"Error caching Gemini response {cache_key}: {str(e)}")

# NEW FUNCTION: Load cached Gemini response
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
            "missing_keywords": ["Gemini API key not configured."],
            "recommendations": ["Set GEMINI_API_KEY environment variable."]
        })

    # Add generation_config to ensure deterministic output
    generation_config = {
        "temperature": 0.0, # CRITICAL FOR CONSISTENCY
        "top_p": 1.0,
        "top_k": 1,
    }

    try:
        model = genai.GenerativeModel('gemini-2.0-flash') # Using flash as specified in original code
        input_text = f"Job Description:\n{job_description}\n\nResume:\n{resume_text}"
        
        # Use generation_config for more deterministic output
        response = model.generate_content([input_text, prompt], generation_config=generation_config)
        
        response_text = response.text
        # Clean markdown code fences from Gemini response
        response_text = re.sub(r'^```json\n|\n```$', '', response_text, flags=re.MULTILINE)
        logger.debug(f"Gemini API raw response: {response_text[:500]}...") # Log more of the response for debugging
        return response_text
    except Exception as e:
        logger.error(f"Error in Gemini API response: {str(e)}")
        return json.dumps({
            "percentage_match": 0.0,
            "missing_keywords": [f"Gemini API call failed: {str(e)}"],
            "recommendations": ["Check Gemini API key, network, and prompt structure."]
        })


def parse_resume(resume_text):
    logger.debug(f"Parsing resume text: {resume_text[:100]}...")
    
    # Clean resume text
    resume_text = clean_resume_text(resume_text)
    doc = nlp(resume_text)
    
    # Initialize Matcher for section detection
    matcher = Matcher(nlp.vocab)
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    
    # Define patterns for section headings (case-insensitive, flexible)
    # Using more robust token patterns for sections
    section_patterns = [
        [{"LOWER": {"IN": ["skills", "technical", "competencies", "extracurricular"]}}, {"LOWER": {"IN": ["skills", "competencies"]}, "OP": "?"}],
        [{"LOWER": {"IN": ["experience", "work", "professional"]}}, {"LOWER": "experience", "OP": "?"}],
        [{"LOWER": "education"}],
        [{"LOWER": "projects"}],
        [{"LOWER": {"IN": ["about", "summary", "profile"]}}]
    ]
    
    matcher.add("SECTION", section_patterns)
    
    # Add soft skill phrases to PhraseMatcher
    soft_skill_phrases = [nlp.make_doc(phrase) for phrase in SOFT_SKILLS]
    phrase_matcher.add("SOFT_SKILLS", soft_skill_phrases)

    # Add technical skill phrases to PhraseMatcher for multi-word skills
    technical_skill_phrases_for_matcher = [nlp.make_doc(phrase) for phrase in TECHNICAL_SKILLS if ' ' in phrase or '-' in phrase]
    if technical_skill_phrases_for_matcher: # Only add if there are multi-word technical skills
        phrase_matcher.add("TECHNICAL_SKILLS_PHRASES", technical_skill_phrases_for_matcher)
    
    # Find section headings and associate content
    sections = {"skills": [], "experience": [], "education": [], "projects": [], "about": []}
    current_section = None
    
    # Process line by line for robust section detection
    lines = [line.strip() for line in resume_text.split('\n') if line.strip()]
    for line in lines:
        line_doc = nlp(line)
        line_matches = matcher(line_doc)
        
        found_section_in_line = False
        for match_id, start, end in line_matches:
            section_text = line_doc[start:end].text.lower()
            if any(s in section_text for s in ["skills", "competencies", "extracurricular"]):
                current_section = "skills"
            elif "experience" in section_text:
                current_section = "experience"
            elif "education" in section_text:
                current_section = "education"
            elif "projects" in section_text:
                current_section = "projects"
            elif any(s in section_text for s in ["about", "summary", "profile"]):
                current_section = "about"
            else:
                # If a section pattern matched but not one of our defined types, ignore it
                continue
            found_section_in_line = True
            logger.debug(f"Detected section: {section_text} -> assigned to {current_section}")
            break # Only take the first section header on a line

        # If no new section header found, add line to current section (if any)
        if not found_section_in_line and current_section:
            sections[current_section].append(line)
        elif not found_section_in_line and not current_section:
            # If no section detected yet, or current_section is None, add to a general bucket or 'about'
            # This handles cases where the resume starts with a summary without a clear heading
            sections["about"].append(line)


    # Extract skills (technical and soft)
    skills = []
    soft_skills = []
    
    # Process skills section for technical skills (single words and common phrases)
    skills_text = " ".join(sections["skills"])
    skills_doc = nlp(skills_text)
    
    # First, try to match exact phrases from TECHNICAL_SKILLS using PhraseMatcher
    for match_id, start, end in phrase_matcher(skills_doc):
        if nlp.vocab.strings[match_id] == "TECHNICAL_SKILLS_PHRASES":
            skills.append(skills_doc[start:end].text.lower())
    
    # Then, iterate through tokens for single-word skills and validation
    for token in skills_doc:
        token_text = token.text.lower()
        if token_text in TECHNICAL_SKILLS:
            skills.append(token_text)

    # Extract technical skills from projects and experience sections
    experience_projects_text = " ".join(sections["projects"] + sections["experience"])
    experience_projects_doc = nlp(experience_projects_text)
    
    # Match multi-word technical skills from experience/projects
    for match_id, start, end in phrase_matcher(experience_projects_doc):
        if nlp.vocab.strings[match_id] == "TECHNICAL_SKILLS_PHRASES":
            skills.append(experience_projects_doc[start:end].text.lower())
    
    # Match single-word technical skills from experience/projects
    for token in experience_projects_doc:
        token_text = token.text.lower()
        if token_text in TECHNICAL_SKILLS:
            skills.append(token_text)
            
    # Extract soft skills from entire resume using PhraseMatcher
    # Applying PhraseMatcher on the whole cleaned text ensures catching skills regardless of section
    doc_cleaned = nlp(resume_text)
    for match_id, start, end in phrase_matcher(doc_cleaned):
        if nlp.vocab.strings[match_id] == "SOFT_SKILLS":
            soft_skills.append(doc_cleaned[start:end].text.lower())
    
    parsed_data = {
        "skills": list(set(skills)), # Use set to remove duplicates
        "soft_skills": list(set(soft_skills)), # Use set to remove duplicates
        "raw_text": resume_text
    }
    
    logger.debug(f"Parsed resume: {parsed_data}")
    return parsed_data

def get_resume_cache_key(resume_text):
    """Generate a cache key based on resume text hash."""
    return hashlib.md5(resume_text.encode('utf-8')).hexdigest()

def cache_resume(parsed_resume, cache_key):
    """Cache parsed resume to disk."""
    cache_path = os.path.join(app.config['CACHE_FOLDER'], f"parsed_resume_{cache_key}.pkl") # Differentiate parsed_resume cache
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(parsed_resume, f)
        logger.debug(f"Cached parsed resume: {cache_key}")
    except Exception as e:
        logger.error(f"Error caching parsed resume {cache_key}: {str(e)}")

def load_cached_resume(cache_key):
    """Load cached resume from disk."""
    cache_path = os.path.join(app.config['CACHE_FOLDER'], f"parsed_resume_{cache_key}.pkl") # Differentiate parsed_resume cache
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
        return None, 0.0, "No valid resume file uploaded.", {"skills": [], "soft_skills": [], "raw_text": ""}, ""

    filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        resume_file.save(filename)
        logger.debug(f"Saved resume file: {filename}")

        resume_text = extract_text(filename)
        if not resume_text.strip():
            logger.error(f"Failed to extract text from {filename} (empty content).")
            return None, 0.0, "Failed to extract readable text from resume.", {"skills": [], "soft_skills": [], "raw_text": ""}, ""

        # Check for cached parsed resume
        parsed_resume_cache_key = get_resume_cache_key(resume_text)
        parsed_resume = load_cached_resume(parsed_resume_cache_key)
        
        if parsed_resume:
            logger.debug(f"Loaded parsed resume from cache for {resume_file.filename}.")
        else:
            parsed_resume = parse_resume(resume_text)
            cache_resume(parsed_resume, parsed_resume_cache_key)
            logger.debug(f"Parsed and cached resume for {resume_file.filename}.")

        # --- NEW: Check for cached Gemini API response ---
        gemini_response_cache_key = get_gemini_response_cache_key(job_description, resume_text, input_prompt)
        gemini_response_raw = load_cached_gemini_response(gemini_response_cache_key)

        if gemini_response_raw:
            logger.debug(f"Loaded Gemini response from cache for {resume_file.filename}.")
        else:
            logger.info(f"Calling Gemini API for {resume_file.filename}...")
            gemini_response_raw = get_gemini_response(job_description, resume_text, input_prompt)
            cache_gemini_response(gemini_response_raw, gemini_response_cache_key)
            logger.debug(f"Cached new Gemini response for {resume_file.filename}.")
        # --- END NEW CACHING ---
        
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
            missing_kw_match = re.search(r'"missing_keywords"\s*:\s*\[([^\]]*)\]', gemini_response_raw)
            if missing_kw_match:
                try:
                    keywords_str = missing_kw_match.group(1).strip()
                    if keywords_str:
                        missing_keywords_list = [kw.strip().strip('"') for kw in keywords_str.split(',') if kw.strip()]
                        # Try to reconstruct basic dicts if possible
                        recommendation_data["missing_keywords"] = [{"keyword": k, "type": "unknown", "importance": "unknown"} for k in missing_keywords_list]
                except Exception:
                    logger.debug(f"Failed to extract keywords from malformed JSON string: {missing_kw_match.group(1)}")
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
        return None, 0.0, error_recommendation_data, {"skills": [], "soft_skills": [], "raw_text": ""}, ""

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
        {
          "percentage_match": <number between 0 and 100, float, precise to one decimal place, e.g., 82.5>,
          "missing_keywords": [
            {"keyword": "<string>", "type": "technical/soft/general", "importance": "critical/important/optional"},
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
            for future in concurrent.futures.as_completed(future_to_resume_file):
                original_file = future_to_resume_file[future]
                # Now `recommendation` is `recommendation_data` (a dict)
                filename, percentage, recommendation_data, parsed_resume, resume_text = future.result() 
                
                # Only add valid results to the list
                if filename:
                    processed_results.append({
                        "filename": filename,
                        "percentage_gemini": percentage, 
                        "recommendation_data": recommendation_data, # Store the full structured data
                        "parsed_resume": parsed_resume,
                        "resume_text": resume_text
                    })
                else:
                    logger.error(f"Failed to process {original_file.filename}. Skipping this resume.")
                    processed_results.append({
                        "filename": original_file.filename or "Unnamed_Resume",
                        "percentage_gemini": 0.0,
                        "recommendation_data": recommendation_data, # This will contain the structured error message
                        "parsed_resume": {"skills": [], "soft_skills": [], "raw_text": ""},
                        "resume_text": ""
                    })

        if not processed_results or all(res["percentage_gemini"] == 0.0 and not res["resume_text"].strip() for res in processed_results):
            logger.warning("No valid resumes processed after all attempts.")
            return render_template('home.html', message="No valid resumes processed. Please check file formats or content.")

        # Prepare texts for TF-IDF vectorization
        all_texts_for_tfidf = [job_description] + [res["resume_text"] for res in processed_results]
        
        tfidf_scores = [0.0] * len(processed_results) # Initialize with zeros
        
        try:
            valid_tfidf_texts = [text for text in all_texts_for_tfidf if text.strip()]
            if len(valid_tfidf_texts) > 1:
                vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.9)
                vectors = vectorizer.fit_transform(valid_tfidf_texts)
                
                job_desc_index_in_valid = -1
                for i, text in enumerate(valid_tfidf_texts):
                    if text == job_description:
                        job_desc_index_in_valid = i
                        break
                
                if job_desc_index_in_valid != -1 and vectors.shape[1] > 0:
                    job_vector = vectors[job_desc_index_in_valid]
                    resume_vectors_for_tfidf = [vectors[i] for i, text in enumerate(valid_tfidf_texts) if text != job_description]
                    
                    if resume_vectors_for_tfidf:
                        similarities = cosine_similarity(job_vector, resume_vectors_for_tfidf).flatten()
                        current_tfidf_idx = 0
                        for i, res in enumerate(processed_results):
                            if res["resume_text"].strip() and current_tfidf_idx < len(similarities):
                                tfidf_scores[i] = round(similarities[current_tfidf_idx] * 100, 2)
                                current_tfidf_idx += 1
                            else:
                                tfidf_scores[i] = 0.0
                    else:
                        logger.warning("No valid resume vectors for TF-IDF similarity calculation.")
                else:
                    logger.warning("TF-IDF vectorization resulted in empty features or job description not found.")
            else:
                logger.warning("Insufficient valid texts for TF-IDF vectorization (need at least 2 non-empty).")
        except Exception as e:
            logger.error(f"Error during TF-IDF vectorization: {str(e)}")

        final_resumes_for_display = []
        # Combine scores using hybrid approach
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

            # Prepare data to pass to template
            final_resumes_for_display.append({
                "filename": res["filename"],
                "score": round(combined_score, 2), 
                "skills": res["parsed_resume"]["skills"],
                "soft_skills": res["parsed_resume"]["soft_skills"],
                "recommendation_data": res["recommendation_data"] # Pass the full structured data
            })

        # Sort top resumes based on the combined percentage match
        top_resumes = sorted(final_resumes_for_display, key=lambda x: x['score'], reverse=True)[:5]

        logger.info(f"Top matching resumes calculated: {[r['filename'] for r in top_resumes]}")
        # RENDER TO NEW TEMPLATE: ats.html
        return render_template(
            'ats.html', # Changed from 'home.html'
            job_description=job_description, # Pass job description for context
            top_resumes=top_resumes
        )

    return render_template('home.html') # This is for initial GET request to /matcher


if __name__ == '__main__':
    # Create upload and cache folders if they don't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        logger.info(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")
    if not os.path.exists(app.config['CACHE_FOLDER']):
        os.makedirs(app.config['CACHE_FOLDER'])
        logger.info(f"Created cache folder: {app.config['CACHE_FOLDER']}")

    app.run(debug=True, host='0.0.0.0')
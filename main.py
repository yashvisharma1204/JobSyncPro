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
app.config['CACHE_FOLDER'] = 'Cache/'

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key="AIzaSyBUHSD621I77-EywekVht9cx1F9evCWNAo")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {str(e)}")
    raise

# Expanded list of soft skills
SOFT_SKILLS = {
    "communication", "teamwork", "leadership", "problem-solving", "adaptability",
    "time management", "collaboration", "creativity", "work ethic", "interpersonal skills",
    "organizational skills", "problem solving", "attention to detail"
}

# Expanded list of technical skills for validation
TECHNICAL_SKILLS = {
    "python", "java", "javascript", "html", "css", "mysql", "mongodb", "pandas",
    "numpy", "matplotlib", "scikit-learn", "power bi", "aws", "azure", "git",
    "github", "flask", "socketio", "pyspark", "databricks", "gemini api", "seaborn",
    "apache spark", "xgboost", "linear regression", "decision tree", "rest apis",
    "html/css", "ci/cd", "angular js", "react", "oracle"
}

def clean_resume_text(resume_text):
    """Clean resume text to remove OCR artifacts and normalize formatting."""
    # Remove repetitive numbers (e.g., "4 4 19 19...")
    resume_text = re.sub(r'\b(\d+\s+)+\d+\b', ' ', resume_text)
    # Remove non-ASCII characters and special symbols
    resume_text = re.sub(r'[^\x00-\x7F]+', ' ', resume_text)
    # Remove email addresses, phone numbers, and social media links
    resume_text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', resume_text)
    resume_text = re.sub(r'\b\+?\d{10,}\b', ' ', resume_text)
    resume_text = re.sub(r'/(linkedin|github|envel)\S*', ' ', resume_text)
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

def get_gemini_response(job_description, resume_text, prompt):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        input_text = f"Job Description:\n{job_description}\n\nResume:\n{resume_text}"
        response = model.generate_content([input_text, prompt])
        response_text = response.text
        # Clean markdown code fences from Gemini response
        response_text = re.sub(r'^```json\n|\n```$', '', response_text, flags=re.MULTILINE)
        logger.debug(f"Gemini API response: {response_text[:100]}...")
        return response_text
    except Exception as e:
        logger.error(f"Error in Gemini API response: {str(e)}")
        return f"Error generating recommendations: {str(e)}"

def parse_resume(resume_text):
    logger.debug(f"Parsing resume text: {resume_text[:100]}...")
    
    # Clean resume text
    resume_text = clean_resume_text(resume_text)
    doc = nlp(resume_text)
    
    # Initialize Matcher for section detection
    matcher = Matcher(nlp.vocab)
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    
    # Define patterns for section headings (case-insensitive, flexible)
    section_patterns = [
        [{"TEXT": {"REGEX": r"(?i)^(skills|technical\s*skills|core\s*competencies|extracurricular\s*skills)\b"}}],
        [{"TEXT": {"REGEX": r"(?i)^(experience|work\s*experience|professional\s*experience)\b"}}],
        [{"TEXT": {"REGEX": r"(?i)^(education)\b"}}],
        [{"TEXT": {"REGEX": r"(?i)^(projects)\b"}}],
        [{"TEXT": {"REGEX": r"(?i)^(about\s*me|summary|profile)\b"}}]
    ]
    
    matcher.add("SECTION", section_patterns)
    
    # Add soft skill phrases to PhraseMatcher
    soft_skill_phrases = [
        "problem solving", "team work", "time management", "interpersonal skills",
        "organizational skills", "attention to detail", "communication", "teamwork",
        "leadership", "problem-solving", "adaptability", "collaboration", "creativity",
        "work ethic"
    ]
    patterns = [nlp.make_doc(phrase) for phrase in soft_skill_phrases]
    phrase_matcher.add("SOFT_SKILLS", patterns)
    
    # Find section headings and associate content
    sections = {"skills": [], "experience": [], "education": [], "projects": [], "about": []}
    current_section = None
    
    # Split text by lines, handling OCR formatting
    lines = [line.strip() for line in resume_text.split('\n') if line.strip()]
    for line in lines:
        line_doc = nlp(line)
        line_matches = matcher(line_doc)
        
        # Check if line is a section heading
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
                current_section = None
            logger.debug(f"Detected section: {section_text}")
            continue
        
        # Add line to current section
        if current_section:
            sections[current_section].append(line)
    
    # Extract skills and soft skills
    skills = []
    soft_skills = []
    
    # Process skills section
    skills_text = " ".join(sections["skills"])
    skills_doc = nlp(skills_text)
    
    # Handle skill formats with validation
    for skill in re.split(r'[,\-\n:;]+', skills_text):
        skill = skill.strip().lower()
        if skill and skill in TECHNICAL_SKILLS:
            skills.append(skill)
    
    # Extract technical skills from projects and experience sections
    projects_text = " ".join(sections["projects"] + sections["experience"])
    projects_doc = nlp(projects_text)
    for token in projects_doc:
        token_text = token.text.lower()
        if token_text in TECHNICAL_SKILLS:
            skills.append(token_text)
    
    # Extract soft skills from entire resume using PhraseMatcher
    for match_id, start, end in phrase_matcher(doc):
        soft_skills.append(doc[start:end].text.lower().replace(" ", "-"))
    
    # Additional soft skill extraction using regex for contextual phrases
    for phrase in soft_skill_phrases:
        if re.search(r'\b' + re.escape(phrase) + r'\b', resume_text, re.IGNORECASE):
            soft_skills.append(phrase.lower().replace(" ", "-"))
    
    parsed_data = {
        "skills": list(set(skills)),
        "soft_skills": list(set(soft_skills)),
        "raw_text": resume_text
    }
    
    logger.debug(f"Parsed resume: {parsed_data}")
    return parsed_data

def get_resume_cache_key(resume_text):
    """Generate a cache key based on resume text hash."""
    return hashlib.md5(resume_text.encode('utf-8')).hexdigest()

def cache_resume(parsed_resume, cache_key):
    """Cache parsed resume to disk."""
    cache_path = os.path.join(app.config['CACHE_FOLDER'], f"{cache_key}.pkl")
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(parsed_resume, f)
        logger.debug(f"Cached resume: {cache_key}")
    except Exception as e:
        logger.error(f"Error caching resume: {str(e)}")

def load_cached_resume(cache_key):
    """Load cached resume from disk."""
    cache_path = os.path.join(app.config['CACHE_FOLDER'], f"{cache_key}.pkl")
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
        logger.debug(f"Loaded cached resume: {cache_key}")
    except Exception:
        return None

def process_resume(resume_file, job_description, input_prompt):
    """Process a single resume file."""
    if not resume_file.filename:
        logger.warning("Empty resume file received")
        return None, 0.0, "Empty resume file.", {"skills": [], "soft_skills": [], "raw_text": ""}, ""

    filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
    try:
        resume_file.save(filename)
        logger.debug(f"Saved resume file: {filename}")
        resume_text = extract_text(filename)
        if not resume_text.strip():
            logger.error(f"Failed to extract text from {filename}")
            return None, 0.0, "Failed to extract resume text.", {"skills": [], "soft_skills": [], "raw_text": ""}, ""

        # Check cache
        cache_key = get_resume_cache_key(resume_text)
        cached_resume = load_cached_resume(cache_key)
        if cached_resume:
            parsed_resume = cached_resume
        else:
            parsed_resume = parse_resume(resume_text)
            cache_resume(parsed_resume, cache_key)

        gemini_response = get_gemini_response(job_description, resume_text, input_prompt)
        try:
            response_json = json.loads(gemini_response)
            percentage = float(response_json.get("percentage_match", 0.0))
            recommendation = json.dumps(response_json, indent=2)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            # Improved regex to match percentage in JSON
            match = re.search(r'"percentage_match"\s*:\s*(\d+\.?\d*)', gemini_response)
            percentage = float(match.group(1)) if match else 0.0
            recommendation = gemini_response + "\nError: Could not parse JSON, fell back to regex."
        return filename, percentage, recommendation, parsed_resume, resume_text
    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}")
        return None, 0.0, f"Error processing {filename}: {str(e)}", {"skills": [], "soft_skills": [], "raw_text": ""}, ""

@app.route("/")
def matchresume():
    return render_template('home.html')

@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form.get('job_description', '').strip()
        resume_files = request.files.getlist('resumes')

        logger.debug(f"Received {len(resume_files)} resume files")
        logger.debug(f"Job description: {job_description[:100]}...")

        if not resume_files or not job_description:
            logger.warning("No resumes or job description provided")
            return render_template('home.html', message="Please upload resumes and enter a job description.")

        resumes = []
        parsed_resumes = []
        recommendations = []
        percentage_matches = []

        # Gemini API prompt for percentage match and recommendations
        input_prompt = """
        You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality. 
        Evaluate the resume against the provided job description. Return the response in JSON format:
        {
          "percentage_match": <number between 0 and 100>,
          "missing_keywords": [<string>, ...],
          "recommendations": [<string>, ...]
        }
        """

        # Process resumes in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_resume = {
                executor.submit(process_resume, resume_file, job_description, input_prompt): resume_file
                for resume_file in resume_files
            }
            for future in concurrent.futures.as_completed(future_to_resume):
                filename, percentage, recommendation, parsed_resume, resume_text = future.result()
                if filename:
                    resumes.append(resume_text)
                    parsed_resumes.append(parsed_resume)
                    percentage_matches.append(percentage)
                    recommendations.append(recommendation)
                else:
                    resumes.append("")
                    parsed_resumes.append(parsed_resume)
                    percentage_matches.append(percentage)
                    recommendations.append(recommendation)

        if not resumes or all(not r.strip() for r in resumes):
            logger.warning("No valid resumes processed")
            return render_template('home.html', message="No valid resumes processed.")

        # Vectorize job description and resumes (fallback scoring)
        try:
            valid_texts = [t for t in [job_description] + resumes if t.strip()]
            if len(valid_texts) >= 2:
                vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.9)
                vectors = vectorizer.fit_transform(valid_texts)
                logger.debug(f"Vectorized {len(valid_texts)} documents. Feature names: {vectorizer.get_feature_names_out()[:10]}...")
                logger.debug(f"Vector shape: {vectors.shape}")
                if vectors.shape[1] == 0:
                    logger.warning("Vectorization produced empty vectors. Check job description and resume content.")
                else:
                    job_vector = vectors[0]
                    resume_vectors = vectors[1:]
                    similarities = cosine_similarity(job_vector, resume_vectors).flatten()
                    tfidf_scores = [round(score * 100, 2) for score in similarities]
                    # Use Gemini percentage if available, else fall back to TF-IDF
                    for i in range(len(resumes)):
                        if percentage_matches[i] == 0.0 and i < len(tfidf_scores):
                            percentage_matches[i] = tfidf_scores[i]
            else:
                logger.warning("Insufficient valid texts for vectorization")
        except Exception as e:
            logger.error(f"Error in vectorization: {str(e)}")

        # Get top 5 resumes based on percentage matches
        top_indices = sorted(range(len(percentage_matches)), key=lambda i: percentage_matches[i], reverse=True)[:5]
        top_resumes = [
            {
                "filename": resume_files[i].filename or f"Resume_{i+1}",
                "score": percentage_matches[i],
                "skills": parsed_resumes[i]["skills"],
                "soft_skills": parsed_resumes[i]["soft_skills"],
                "recommendations": recommendations[i]
            }
            for i in top_indices if i < len(resume_files) and i < len(parsed_resumes)
        ]

        logger.debug(f"Top resumes: {top_resumes}")
        return render_template(
            'home.html',
            message="Top matching resumes:",
            top_resumes=top_resumes
        )

    return render_template('home.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['CACHE_FOLDER']):
        os.makedirs(app.config['CACHE_FOLDER'])
    app.run(debug=True)
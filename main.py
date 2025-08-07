from flask import Flask, request, render_template, session, redirect, url_for, jsonify
import os
import json
import concurrent.futures
import uuid
import shutil
from functools import wraps
from datetime import timedelta
import logging
from prep import prep_bp
from flask_session import Session
import firebase_admin
from firebase_admin import credentials, auth

from utils.text_extraction import extract_text
from utils.resume_parser import parse_resume
from utils.gemini_api import get_gemini_response
from utils.scoring_utils import analyze_resume_quality, calculate_tfidf_score, calculate_combined_score
from utils.cache_utils import (
    get_resume_cache_key, load_cached_resume, cache_resume,
    get_gemini_response_cache_key, load_cached_gemini_response, cache_gemini_response
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads/'
app.config['CACHE_FOLDER'] = 'Cache/'


app.config['SECRET_KEY'] = 'a-very-long-and-super-secret-random-string-for-your-app'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    service_account_path = os.path.join(BASE_DIR, 'firebase-service-account.json')
    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(cred)
    logging.info("Firebase Admin SDK initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Firebase Admin SDK. Make sure 'firebase-service-account.json' is present. Error: {e}")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_uid' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

def process_resume(resume_file, job_description, input_prompt):
    """Process a single resume file by orchestrating calls to utility modules."""
    if not resume_file or not resume_file.filename:
        logger.warning("Empty or no resume file received in process_resume.")
        return None, 0.0, {"percentage_match": 0.0, "missing_keywords": [], "recommendations": [], "quality_feedback": {}}, {"skills": [], "soft_skills": [], "raw_text": ""}, ""

    filename_for_processing = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_processing', resume_file.filename)
    try:
        temp_processing_dir = os.path.dirname(filename_for_processing)
        os.makedirs(temp_processing_dir, exist_ok=True)
        resume_file.save(filename_for_processing)
        logger.debug(f"Copied resume file for processing: {filename_for_processing}")

        resume_text = extract_text(filename_for_processing)
        if not resume_text.strip():
            logger.error(f"Failed to extract text from {filename_for_processing} (empty content).")
            return None, 0.0, {"percentage_match": 0.0, "missing_keywords": [], "recommendations": [], "quality_feedback": {}}, {"skills": [], "soft_skills": [], "raw_text": ""}, ""

        parsed_resume_cache_key = get_resume_cache_key(resume_text)
        parsed_resume = load_cached_resume(parsed_resume_cache_key, app.config['CACHE_FOLDER'])

        if parsed_resume:
            logger.debug(f"Loaded parsed resume from cache for {resume_file.filename}.")
        else:
            parsed_resume = parse_resume(resume_text)
            cache_resume(parsed_resume, parsed_resume_cache_key, app.config['CACHE_FOLDER'])
            logger.debug(f"Parsed and cached resume for {resume_file.filename}.")

        local_quality_feedback = analyze_resume_quality(parsed_resume["raw_text"], parsed_resume["sections"])

        gemini_response_cache_key = get_gemini_response_cache_key(job_description, resume_text, input_prompt)
        gemini_response_raw = load_cached_gemini_response(gemini_response_cache_key, app.config['CACHE_FOLDER'])

        if gemini_response_raw:
            logger.debug(f"Loaded Gemini response from cache for {resume_file.filename}.")
        else:
            logger.info(f"Calling Gemini API for {resume_file.filename}...")
            gemini_response_raw = get_gemini_response(job_description, resume_text, input_prompt)
            cache_gemini_response(gemini_response_raw, gemini_response_cache_key, app.config['CACHE_FOLDER'])
            logger.debug(f"Cached new Gemini response for {resume_file.filename}.")

        percentage = 0.0
        recommendation_data = {
            "percentage_match": 0.0, "missing_keywords": [], "recommendations": [], "quality_feedback": {}
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
            percentage = 0.0
            recommendation_data["percentage_match"] = 0.0
            recommendation_data["recommendations"] = [
                "CRITICAL ERROR: The AI did not return a valid JSON response and could not be scored.",
                "See the raw AI output below for details.",
                f"Raw AI Output: {gemini_response_raw}"
            ]
            recommendation_data["quality_feedback"] = {"error": "Could not parse AI response."}

        final_quality_feedback = {}
        final_quality_feedback.update(local_quality_feedback)
        final_quality_feedback.update(recommendation_data["quality_feedback"])
        recommendation_data["quality_feedback"] = final_quality_feedback

        return resume_file.filename, percentage, recommendation_data, parsed_resume, resume_text
    except Exception as e:
        logger.error(f"Critical error processing file {resume_file.filename}: {str(e)}")
        error_recommendation_data = {
            "percentage_match": 0.0,
            "missing_keywords": [],
            "recommendations": [f"Critical error processing {resume_file.filename}: {str(e)}", "Please check server logs."],
            "quality_feedback": {"system_error": f"Critical processing error: {str(e)}"}
        }
        return None, 0.0, error_recommendation_data, {"skills": [], "soft_skills": [], "raw_text": ""}, ""
    finally:
        try:
            if os.path.exists(filename_for_processing):
                os.remove(filename_for_processing)
                logger.debug(f"Removed temporary processing file: {filename_for_processing}")
        except OSError as e:
            logger.error(f"Error removing file {filename_for_processing}: {e}")


class MockFileStorage:
    """A helper class to mimic a Flask FileStorage object from a file path."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)

    def save(self, dst):
        """Copies the file to the destination, mimicking the save method."""
        try:
            shutil.copy(self.filepath, dst)
        except Exception as e:
            logger.error(f"MockFileStorage could not copy {self.filepath} to {dst}: {e}")
            raise

@app.route("/")
def index():
    return render_template('main.html')

@app.route("/login")
def login_page():
    if 'user_uid' in session:
        return redirect(url_for('home_page'))
    return render_template('auth.html')

@app.route("/home")
@login_required
def home_page():
    session.pop('job_id', None)
    session.pop('job_description', None)
    session.pop('resume_filenames', None)
    return render_template('home.html')

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/verify-token', methods=['POST'])
def verify_token():
    try:
        data = request.get_json()
        token = data.get('token')
        remember_me = data.get('remember', False) 

        if not token:
            return jsonify({'status': 'error', 'message': 'Token is missing'}), 400

        decoded_token = auth.verify_id_token(token)
        session['user_uid'] = decoded_token['uid']
        session['user_email'] = decoded_token.get('email', 'N/A')

        if remember_me:
            session.permanent = True
        
        logger.info(f"User authenticated: {session['user_email']} ({session['user_uid']}), Remember Me: {remember_me}")
        return jsonify({'status': 'success', 'uid': decoded_token['uid']}), 200
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 401


@app.route('/matcher', methods=['POST'])
@login_required
def loader():
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
    job_id = session.get('job_id')
    job_description = session.get('job_description')
    resume_filenames = session.get('resume_filenames')

    if not all([job_id, job_description, resume_filenames]):
        logger.error("Results page accessed without job data in session. Redirecting to home.")
        return redirect(url_for('index'))

    job_upload_path = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    resume_files = [MockFileStorage(os.path.join(job_upload_path, fname)) for fname in resume_filenames]

    logger.info(f"Processing job {job_id} retrieved from session with {len(resume_files)} resumes.")

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

    processed_results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_resume_file = {
            executor.submit(process_resume, resume_file, job_description, input_prompt): resume_file
            for resume_file in resume_files
        }
        for future in concurrent.futures.as_completed(future_to_resume_file):
            original_file = future_to_resume_file[future]
            try:
                filename, percentage, recommendation_data, parsed_resume, resume_text = future.result()
                processed_results.append({
                    "filename": filename or original_file.filename,
                    "percentage_gemini": percentage,
                    "recommendation_data": recommendation_data,
                    "parsed_resume": parsed_resume,
                    "resume_text": resume_text
                })
            except Exception as exc:
                logger.error(f'{original_file.filename} generated an exception: {exc}')


    if not processed_results or all(res["percentage_gemini"] == 0.0 and not res["resume_text"].strip() for res in processed_results):
        logger.warning("No valid resumes processed after all attempts.")
        return render_template('home.html', message="No valid resumes processed. Please check file formats or content.")

    final_resumes_for_display = []
    GEMINI_COMPREHENSIVE_WEIGHT = 0.85
    TFIDF_WEIGHT = 0.15

    for res in processed_results:
        gemini_match_score = res["percentage_gemini"]
        tfidf_score = calculate_tfidf_score(job_description, res["resume_text"])
        combined_score = calculate_combined_score(gemini_match_score, tfidf_score, GEMINI_COMPREHENSIVE_WEIGHT, TFIDF_WEIGHT)

        final_resumes_for_display.append({
            "filename": res["filename"],
            "score": round(combined_score, 2),
            "skills": res["parsed_resume"]["skills"],
            "soft_skills": res["parsed_resume"]["soft_skills"],
            "recommendation_data": res["recommendation_data"],
            "resume_text": res["resume_text"]  # <-- MODIFICATION: Pass resume text
        })
    
    top_resumes = sorted(final_resumes_for_display, key=lambda x: x['score'], reverse=True)[:5]
    logger.info(f"Top matching resumes calculated for job {job_id}: {[r['filename'] for r in top_resumes]}")

    try:
        shutil.rmtree(job_upload_path)
        logger.info(f"Cleaned up temporary directory: {job_upload_path}")
    except OSError as e:
        logger.error(f"Error removing temporary directory {job_upload_path}: {e}")
    
    session.pop('job_id', None)
    session.pop('job_description', None)
    session.pop('resume_filenames', None)

    return render_template(
        'ats.html',
        job_description=job_description,
        top_resumes=top_resumes
    )

app.register_blueprint(prep_bp)
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['CACHE_FOLDER']):
        os.makedirs(app.config['CACHE_FOLDER'])
    
    app.run(debug=True, host='0.0.0.0')
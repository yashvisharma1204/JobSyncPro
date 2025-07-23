from flask import Flask, request, render_template
import os
import logging
from utils.text_extraction import extract_text
from utils.resume_parser import parse_resume
from utils.cache_utils import get_resume_cache_key, cache_resume, load_cached_resume, get_gemini_response_cache_key, cache_gemini_response, load_cached_gemini_response
from utils.gemini_api import get_gemini_response
from utils.scoring_utils import calculate_tfidf_scores, combine_scores
import concurrent.futures
import json
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads/'
app.config['CACHE_FOLDER'] = 'Cache/'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        recommendation_data = {
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
                            missing_keywords_list = [kw.strip().strip('"') for kw in re.split(r',\s*(?="|\b)', keywords_str) if kw.strip()]
                            recommendation_data["missing_keywords"] = [{"keyword": k, "type": "unknown", "importance": "unknown"} for k in missing_keywords_list]
                except Exception as ex:
                    logger.debug(f"Failed to extract keywords from malformed JSON string '{keywords_str}' using fallback: {ex}")
                    pass
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
            pass
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

        processed_results = []
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

        tfidf_scores = calculate_tfidf_scores(job_description, processed_results)
        final_resumes = combine_scores(processed_results, tfidf_scores)
        top_resumes = sorted(final_resumes, key=lambda x: x['score'], reverse=True)[:5]
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
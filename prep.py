
from flask import Blueprint, render_template, request, session, redirect, url_for
import logging
import json
import os
import uuid
from utils.text_extraction import extract_text
from utils.gemini_api import get_gemini_response

# Create a Blueprint, which is a way to organize a group of related routes
prep_bp = Blueprint('prep', __name__)
logger = logging.getLogger(__name__)

@prep_bp.route('/start-interview', methods=['POST'])
def start_interview():
    """
    This route receives the JD and resume, generates interview questions,
    and redirects the user to the interview page.
    """
    job_description = request.form.get('job_description', '').strip()
    resume_files = request.files.getlist('resumes')
    resume_text = ""

    temp_file_path = None
    try:
        if resume_files and resume_files[0].filename:
            resume_file = resume_files[0]
            
            upload_folder = 'Uploads'
            os.makedirs(upload_folder, exist_ok=True)
            temp_filename = str(uuid.uuid4()) + "_" + resume_file.filename
            temp_file_path = os.path.join(upload_folder, temp_filename)
            
            resume_file.save(temp_file_path)
            resume_text = extract_text(temp_file_path)
            logger.info(f"Extracted text from uploaded file: {resume_file.filename}")
        else:
            resume_text = request.form.get('resume_text', '').strip()

        if not job_description or not resume_text:
            return redirect(url_for('home.home_page', message="Job Description or Resume was missing."))

        question_generation_prompt = f"""
        Based on the following job description and resume text, act as a senior hiring manager. Your task is to generate exactly 10 relevant interview questions.

        **CRITICAL INSTRUCTIONS ON QUESTION STRUCTURE:**
        1.  **The first question MUST be a comprehensive "Tell me about yourself" question.** This question should prompt the candidate to structure their answer by summarizing their professional background, key technical skills (hard skills), collaborative abilities (soft skills), and then briefly introduce one or two of their most relevant projects mentioned in the resume.
        2.  **The remaining 9 questions** should be a mix of technical and behavioral questions derived from the candidate's specific projects, listed skills, and the requirements in the job description.

        **CRITICAL INSTRUCTIONS ON OUTPUT FORMAT:**
        Your entire response MUST be a single, valid JSON object, with a single key "questions" that holds an array of 10 question strings. Do not add any other text or formatting.
        Example: {{"questions": ["Tell me about yourself, touching on your skills, projects, and how they relate to this role.", "In your project X, how did you handle...", ...]}}

        **Job Description:**
        {job_description}

        **Resume Text:**
        {resume_text}
        """

        response_raw = get_gemini_response(job_description, resume_text, question_generation_prompt)
        if not response_raw:
             raise ValueError("AI API returned an empty response, likely due to an error (e.g., rate limiting).")

        response_json = json.loads(response_raw)
        questions = response_json.get("questions", [])

        if not questions or len(questions) < 5:
            raise ValueError("AI did not generate enough questions.")

        session['interview_questions'] = questions
        session['interview_context'] = {
            'job_description': job_description,
            'resume_text': resume_text
        }
        return redirect(url_for('prep.interview_page'))

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to generate interview questions: {e}")
        return redirect(url_for('home.home_page', message="Could not generate interview questions. Please try again."))
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Removed temporary file: {temp_file_path}")

@prep_bp.route('/interview')
def interview_page():
    """Displays the interview interface with questions from the session."""
    questions = session.get('interview_questions')
    if not questions:
        return redirect(url_for('home.home_page'))
    
    # Pass any message to the template if it exists
    message = request.args.get('message')
    return render_template('interview.html', questions=questions, message=message)

@prep_bp.route('/get-interview-results', methods=['POST'])
def get_interview_results():
    """Receives user's answers, evaluates them with Gemini, and shows the results."""
    questions = session.get('interview_questions')
    context = session.get('interview_context')
    
    if not all([questions, context]):
        return redirect(url_for('home.home_page', message="Session expired or data was lost. Please start over."))

    # --- MODIFICATION START ---
    answers = [request.form.get(f'answer_{i}', '').strip() for i in range(len(questions))]

    qa_pairs_text = ""
    for i, q in enumerate(questions):
        answer_text = answers[i] if answers[i] else "[SKIPPED]"
        qa_pairs_text += f"Question {i+1}: {q}\nAnswer {i+1}: {answer_text}\n\n"

    evaluation_prompt = f"""
    You are an expert interview coach. Your task is to evaluate a candidate's answers to interview questions, based on their resume and the job description they are applying for.

    Provide an overall score out of 10 and constructive feedback for each answer.

    **CRITICAL INSTRUCTION FOR SKIPPED QUESTIONS:**
    If an answer is marked as "[SKIPPED]", you MUST give it a score of 0 and your feedback for it must be "The candidate did not provide an answer to this question."

    **CRITICAL**: Your entire response MUST be a single, valid JSON object. Do not add any other text.
    The JSON structure MUST be:
    {{
      "overall_score": <number out of 10>,
      "overall_feedback": "<string>",
      "answer_evaluations": [
        {{
          "question": "<The original question>",
          "answer": "<The user's answer>",
          "feedback": "<Your constructive feedback on this answer>",
          "score": <number out of 10 for this answer>
        }},
        ...
      ]
    }}

    **Job Description:**
    {context['job_description']}

    **Candidate's Resume Summary:**
    {context['resume_text'][:1000]}

    **Questions and Answers to Evaluate:**
    {qa_pairs_text}
    """

    try:
        response_raw = get_gemini_response(context['job_description'], qa_pairs_text, evaluation_prompt)
        results = json.loads(response_raw)
        
        # Ensure the original blank answer is shown on the results page, not '[SKIPPED]'
        for i, eval_item in enumerate(results.get("answer_evaluations", [])):
            if eval_item.get('answer') == "[SKIPPED]":
                eval_item['answer'] = ""

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to evaluate interview answers: {e}")
        # Redirect back to the interview page with an error message
        return redirect(url_for('prep.interview_page', message="Could not evaluate interview answers. Please try again."))

    session.pop('interview_questions', None)
    session.pop('interview_context', None)
    
    return render_template('interview_results.html', results=results)

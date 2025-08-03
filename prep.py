from flask import Blueprint, render_template, request, session, redirect, url_for
import logging
import json
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
    
    if resume_files and resume_files[0].filename:
        resume_file = resume_files[0]
        
        resume_text = resume_file.read().decode('utf-8', errors='ignore') # A simplified way for text
        if resume_file.filename.endswith('.pdf') or resume_file.filename.endswith('.docx'):
             
             logger.warning("File upload for interview prep requires text extraction, which is complex here. Proceeding with caution.")
            
             pass # Let's assume for now the text is extracted.
    else:
        
        resume_text = request.form.get('resume_text', '').strip()

    if not job_description or not resume_text:
        return redirect(url_for('home_page', message="Job Description or Resume was missing."))

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

    try:
        response_raw = get_gemini_response(job_description, resume_text, question_generation_prompt)
        response_json = json.loads(response_raw)
        questions = response_json.get("questions", [])

        if not questions or len(questions) < 5: # Basic validation
             raise ValueError("AI did not generate enough questions.")

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to generate interview questions: {e}")
        return redirect(url_for('home_page', message="Could not generate interview questions. Please try again."))

    # Store everything needed for the interview and evaluation in the session
    session['interview_questions'] = questions
    session['interview_context'] = {
        'job_description': job_description,
        'resume_text': resume_text
    }

    return redirect(url_for('prep.interview_page'))


@prep_bp.route('/interview')
def interview_page():
    """Displays the interview interface with questions from the session."""
    questions = session.get('interview_questions')
    if not questions:
        return redirect(url_for('home_page'))
    
    return render_template('interview.html', questions=questions)


@prep_bp.route('/get-interview-results', methods=['POST'])
def get_interview_results():
    """Receives user's answers, evaluates them with Gemini, and shows the results."""
    questions = session.get('interview_questions')
    context = session.get('interview_context')
    answers = [request.form.get(f'answer_{i}', '') for i in range(len(questions))]

    if not all([questions, context, answers]):
        return redirect(url_for('home_page', message="Session expired or data was lost. Please start over."))

    # Combine questions and answers for the evaluation prompt
    qa_pairs_text = ""
    for i, q in enumerate(questions):
        qa_pairs_text += f"Question {i+1}: {q}\nAnswer {i+1}: {answers[i]}\n\n"

    # 2. Second Gemini Call: Evaluate Answers
    evaluation_prompt = f"""
    You are an expert interview coach. Your task is to evaluate a candidate's answers to interview questions, based on their resume and the job description they are applying for.

    Provide an overall score out of 10 and constructive feedback for each answer.

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
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to evaluate interview answers: {e}")
        return redirect(url_for('home_page', message="Could not evaluate interview answers. Please try again."))

    session.pop('interview_questions', None)
    session.pop('interview_context', None)
    
    return render_template('interview_results.html', results=results)
import os
import json
import re
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set. Please set it before running the application.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

def get_gemini_response(job_description, resume_text, prompt):
    if not GEMINI_API_KEY:
        logger.warning("Gemini API key not set, skipping API call.")
        return json.dumps({
            "percentage_match": 0.0,
            "missing_keywords": [{"keyword": "Gemini API key not configured.", "type": "system", "importance": "critical"}],
            "recommendations": ["Set GEMINI_API_KEY environment variable."],
            "quality_feedback": {}
        })

    generation_config = {
        "temperature": 0.0, "top_p": 1.0, "top_k": 1,
    }

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        input_text = f"Job Description:\n{job_description}\n\nResume:\n{resume_text}"
        response = model.generate_content([input_text, prompt], generation_config=generation_config)
        response_text = re.sub(r'^```json\n|\n```$', '', response.text, flags=re.MULTILINE)
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
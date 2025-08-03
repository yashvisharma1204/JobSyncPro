import os
import pickle
import hashlib
import logging

logger = logging.getLogger(__name__)

def get_gemini_response_cache_key(job_description, resume_text, prompt):
    combined_input = job_description + "|||" + resume_text + "|||" + prompt
    return hashlib.md5(combined_input.encode('utf-8')).hexdigest()

def cache_gemini_response(gemini_response, cache_key, cache_folder):
    cache_path = os.path.join(cache_folder, f"gemini_{cache_key}.pkl")
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(gemini_response, f)
        logger.debug(f"Cached Gemini response: {cache_key}")
    except Exception as e:
        logger.error(f"Error caching Gemini response {cache_key}: {str(e)}")

def load_cached_gemini_response(cache_key, cache_folder):
    cache_path = os.path.join(cache_folder, f"gemini_{cache_key}.pkl")
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
    return hashlib.md5(resume_text.encode('utf-8')).hexdigest()

def cache_resume(parsed_resume, cache_key, cache_folder):
    cache_path = os.path.join(cache_folder, f"parsed_resume_{cache_key}.pkl")
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(parsed_resume, f)
        logger.debug(f"Cached parsed resume: {cache_key}")
    except Exception as e:
        logger.error(f"Error caching parsed resume {cache_key}: {str(e)}")

def load_cached_resume(cache_key, cache_folder):
    cache_path = os.path.join(cache_folder, f"parsed_resume_{cache_key}.pkl")
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        else:
            return None
    except Exception as e:
        logger.error(f"Error loading cached parsed resume {cache_key}: {str(e)}")
        return None
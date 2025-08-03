import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

def analyze_resume_quality(resume_text, sections_content_dict):
    quality_feedback = {
        "grammar_issues": ["Grammar and spelling analysis is primarily provided by the AI (Gemini) for advanced contextual review."],
        "structural_suggestions": [],
        "conciseness_suggestions": [],
    }
    return quality_feedback

def calculate_tfidf_score(job_description, resume_text):
    """Calculates the TF-IDF cosine similarity score between two texts."""
    tfidf_score = 0.0
    if resume_text.strip() and job_description.strip():
        try:
            vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.9)
            vectors = vectorizer.fit_transform([job_description, resume_text])

            if vectors.shape[1] > 0:
                similarity = cosine_similarity(vectors[0], vectors[1]).flatten()[0]
                tfidf_score = round(similarity * 100, 2)
                logger.debug(f"TF-IDF score calculated: {tfidf_score}%")
            else:
                logger.debug("TF-IDF vectorization resulted in empty features.")
        except Exception as e:
            logger.error(f"Error during TF-IDF calculation: {str(e)}")
    return tfidf_score

def calculate_combined_score(gemini_score, tfidf_score, gemini_weight, tfidf_weight):
    """Calculates the weighted average of Gemini and TF-IDF scores."""
    return (gemini_score * gemini_weight) + (tfidf_score * tfidf_weight)
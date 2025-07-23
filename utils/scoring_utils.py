from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

def calculate_tfidf_scores(job_description, processed_results):
    all_texts_for_tfidf_input = []
    original_indices_map = []
    job_desc_tfidf_idx = -1

    if job_description.strip():
        all_texts_for_tfidf_input.append(job_description)
        job_desc_tfidf_idx = 0
    for i, res in enumerate(processed_results):
        if res["resume_text"].strip():
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
            logger.warning("Insufficient valid texts for TF-IDF vectorization (need job description and at least one resume).")
    except Exception as e:
        logger.error(f"Error during TF-IDF vectorization: {str(e)}")
    return tfidf_scores

def combine_scores(processed_results, tfidf_scores, gemini_weight=0.7, tfidf_weight=0.3):
    final_resumes = []
    for i, res in enumerate(processed_results):
        gemini_score = res["percentage_gemini"]
        tfidf_score = tfidf_scores[i]
        combined_score = 0.0
        if gemini_score > 0.0 and tfidf_score > 0.0:
            combined_score = (gemini_score * gemini_weight) + (tfidf_score * tfidf_weight)
        elif gemini_score > 0.0:
            combined_score = gemini_score
        elif tfidf_score > 0.0:
            combined_score = tfidf_score
        final_resumes.append({
            "filename": res["filename"],
            "score": round(combined_score, 2),
            "skills": res["parsed_resume"]["skills"],
            "soft_skills": res["parsed_resume"]["soft_skills"],
            "recommendation_data": res["recommendation_data"]
        })
    return final_resumes
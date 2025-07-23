import re
import spacy
from spacy.matcher import Matcher, PhraseMatcher
import logging

logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {str(e)}. Please run 'python -m spacy download en_core_web_sm'")
    raise

SOFT_SKILLS = {
    "communication", "teamwork", "leadership", "problem-solving", "adaptability",
    "time management", "collaboration", "creativity", "work ethic", "interpersonal skills",
    "organizational skills", "critical thinking", "negotiation", "conflict resolution",
    "emotional intelligence", "decision making", "presentation skills", "active listening",
    "attention to detail", "analytical skills"
}

TECHNICAL_SKILLS = {
    "python", "java", "javascript", "html", "css", "mysql", "mongodb", "pandas",
    "numpy", "matplotlib", "scikit-learn", "power bi", "aws", "azure", "git",
    "github", "flask", "socketio", "pyspark", "databricks", "gemini api", "seaborn",
    "apache spark", "xgboost", "linear regression", "decision tree", "rest apis",
    "html/css", "ci/cd", "angular js", "react", "oracle", "machine learning",
    "deep learning", "natural language processing", "data analysis", "data visualization",
    "cloud computing", "devops", "kubernetes", "docker", "sql", "excel", "tableau",
    "r", "c++", "c#", "scala", "go", "spring boot", "node.js", "typescript",
    "data analysis and visualisation", "git and github", "rest apis", "apache spark",
    "linear regression", "decision tree", "aws lambda", "aws glue", "aws iam",
    "next.js", "react.js", "generative ai", "tkinter", "socket.io", "num py",
    "scikit learn", "power bi"
}

def clean_resume_text(resume_text):
    """Clean resume text to remove OCR artifacts and normalize formatting."""
    resume_text = re.sub(r'\b(\d+\s+)+\d+\b', ' ', resume_text)
    resume_text = re.sub(r'[^\x00-\x7F]+', ' ', resume_text)
    resume_text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', resume_text)
    resume_text = re.sub(r'\b\+?\d{10,}\b', ' ', resume_text)
    resume_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', resume_text, flags=re.IGNORECASE)
    resume_text = re.sub(r'[•●■▪]', ' ', resume_text)
    resume_text = re.sub(r'\s+', ' ', resume_text.strip())
    return resume_text

def parse_resume(resume_text):
    logger.debug(f"--- Starting parse_resume ---")
    logger.debug(f"Initial resume text received: {resume_text[:200]}...")

    resume_text = clean_resume_text(resume_text)
    logger.debug(f"Cleaned resume text: {resume_text[:200]}...")
    doc = nlp(resume_text)

    matcher = Matcher(nlp.vocab)
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    section_patterns = [
        [{"LOWER": {"IN": ["summary", "about me", "profile"]}}],
        [{"LOWER": {"IN": ["skills", "technical skills", "core competencies", "extracurricular skills", "technologies", "tech skills"]}}],
        [{"LOWER": {"IN": ["experience", "work experience", "professional experience"]}}],
        [{"LOWER": "education"}],
        [{"LOWER": "projects"}],
        [{"LOWER": {"IN": ["achievements", "certifications", "additional information", "awards", "languages"]}}]
    ]
    matcher.add("SECTION", section_patterns)

    soft_skill_phrases = [nlp.make_doc(phrase) for phrase in SOFT_SKILLS]
    phrase_matcher.add("SOFT_SKILLS", soft_skill_phrases)

    sections = {
        "summary": [], "skills": [], "experience": [],
        "education": [], "projects": [], "additional_info": []
    }
    current_section = None
    lines = [line.strip() for line in resume_text.split('\n') if line.strip()]
    logger.debug(f"Total {len(lines)} non-empty lines after initial split.")

    for line_idx, line in enumerate(lines):
        line_doc = nlp(line)
        line_matches = matcher(line_doc)
        found_section_in_line = False
        for match_id, start, end in line_matches:
            section_text = line_doc[start:end].text.lower()
            if any(s in section_text for s in ["summary", "about me", "profile"]):
                current_section = "summary"
            elif any(s in section_text for s in ["skills", "technical skills", "core competencies", "extracurricular skills", "technologies", "tech skills"]):
                current_section = "skills"
            elif "experience" in section_text:
                current_section = "experience"
            elif "education" in section_text:
                current_section = "education"
            elif "projects" in section_text:
                current_section = "projects"
            elif any(s in section_text for s in ["achievements", "certifications", "additional information", "awards", "languages"]):
                current_section = "additional_info"
            else:
                continue
            found_section_in_line = True
            logger.debug(f"L{line_idx}: Detected section: '{section_text}' -> assigned to {current_section}")
            break
        if not found_section_in_line and current_section:
            sections[current_section].append(line)
        elif not found_section_in_line and not current_section:
            sections["summary"].append(line)

    logger.debug(f"Sections identified: { {k: len(v) for k, v in sections.items()} }")
    logger.debug(f"Content of 'skills' section: {' '.join(sections['skills'])[:200]}...")

    skills = []
    soft_skills = []
    if sections["skills"]:
        skills_text_from_section = " ".join(sections["skills"]).lower()
        logger.debug(f"Attempting to extract skills from dedicated 'skills' section: '{skills_text_from_section}'")
        tech_matcher_for_section = PhraseMatcher(nlp.vocab, attr="LOWER")
        tech_phrases_for_section = [nlp.make_doc(s) for s in TECHNICAL_SKILLS]
        tech_matcher_for_section.add("EXACT_TECH_SKILLS_SECTION", tech_phrases_for_section)
        skills_doc_section = nlp(skills_text_from_section)
        for match_id, start, end in tech_matcher_for_section(skills_doc_section):
            matched_skill = skills_doc_section[start:end].text.lower()
            if matched_skill not in skills:
                skills.append(matched_skill)
                logger.debug(f"Found exact tech skill in dedicated section: '{matched_skill}'")
        words_in_skills_section = re.findall(r'\b[a-zA-Z0-9+#.-]+\b', skills_text_from_section)
        for word in words_in_skills_section:
            word_lower = word.strip().lower()
            if word_lower in TECHNICAL_SKILLS and word_lower not in skills:
                skills.append(word_lower)
                logger.debug(f"Found single word tech skill in dedicated section: '{word_lower}'")
    else:
        logger.debug("Dedicated 'skills' section not found or is empty.")

    full_resume_doc = nlp(resume_text)
    logger.debug("Scanning entire document for technical skills (contextual search)...")
    all_tech_phrases = [nlp.make_doc(s) for s in TECHNICAL_SKILLS]
    phrase_matcher.add("ALL_TECHNICAL_SKILLS_GLOBAL", all_tech_phrases)
    for match_id, start, end in phrase_matcher(full_resume_doc):
        if nlp.vocab.strings[match_id] == "ALL_TECHNICAL_SKILLS_GLOBAL":
            matched_skill = full_resume_doc[start:end].text.lower()
            if matched_skill not in skills:
                skills.append(matched_skill)
                logger.debug(f"Found contextual tech skill in document: '{matched_skill}'")
    for token in full_resume_doc:
        token_text = token.text.lower()
        if token_text in TECHNICAL_SKILLS and token_text not in skills:
            skills.append(token_text)
            logger.debug(f"Found single word tech skill in document (token-level): '{token_text}'")

    logger.debug("Scanning entire document for soft skills...")
    for match_id, start, end in phrase_matcher(doc):
        if nlp.vocab.strings[match_id] == "SOFT_SKILLS":
            matched_soft_skill = doc[start:end].text.lower()
            if matched_soft_skill not in soft_skills:
                soft_skills.append(matched_soft_skill)
                logger.debug(f"Matched soft skill: {matched_soft_skill}")

    skills = sorted(list(set(skills)))
    soft_skills = sorted(list(set(soft_skills)))
    parsed_data = {
        "skills": skills,
        "soft_skills": soft_skills,
        "raw_text": resume_text
    }
    logger.debug(f"--- Finished parse_resume ---")
    logger.debug(f"Parsed resume FINAL technical skills: {parsed_data['skills']}")
    logger.debug(f"Parsed resume FINAL soft skills: {parsed_data['soft_skills']}")
    return parsed_data
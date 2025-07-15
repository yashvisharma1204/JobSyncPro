# JobSyncPro
this project is in progress
the development phase(with react and next js) of this project being going on this repositary : https://github.com/punitkumar4871/Resume_interview_matcher

# Resume ATS Scoring and Interview System
### Overview
`This project consists of two main components:`

Resume ATS Scoring System: Analyzes resumes and matches them to job descriptions using NLP techniques to calculate an Applicant Tracking System (ATS) score. It extracts keywords from job descriptions using SpaCy and provides tailored recommendations to improve resume alignment with job requirements using the Gemini API.
AI-Powered Interview System: Parses resumes using GenAI to identify key sections (e.g., projects, skills, experience), generates relevant interview questions, allows users to submit answers, evaluates responses, and provides a performance score.

# Features
### Resume ATS Scoring System

`Keyword Extraction`: Utilizes SpaCy to extract relevant keywords and phrases from job descriptions.
Resume Matching: Compares resume content with job description keywords to compute an ATS compatibility score.
Recommendations: Leverages the Gemini API to provide actionable suggestions for improving resume content to better align with job requirements.

### Interview System

`Resume Parsing:` Uses GenAI to parse resumes and identify critical sections such as projects, skills, and work experience.
Question Generation: Automatically generates interview questions based on parsed resume content and job requirements.
Answer Evaluation: Allows users to submit answers to generated questions, evaluates responses using GenAI, and provides a performance score with feedback.

#### Technologies Used

`Python:` Core programming language for the project.
`SpaCy:` NLP library for keyword extraction from job descriptions.
`Gemini API:` Used for generating resume improvement recommendations.
`GenAI:` Powers resume parsing, question generation, and answer evaluation.

# Installation

`Clone the repository:git clone https://github.com/yashvisharma1204/JobSyncPro.git`


`Navigate to the project directory:cd JobSyncPro`


`Install dependencies:pip install -r requirements.txt`


`Install SpaCy model:python -m spacy download en_core_web_sm`


### Set up API keys:
`Obtain a Gemini API key and add it to a .env file as GEMINI_API_KEY.`
`Configure any additional API keys or credentials required for GenAI.`



## Usage
#### ATS Scoring

Prepare a job description and resume in text or PDF format.

### Interview System

Answer the generated questions through the provided interface (CLI or web-based, depending on implementation).
Receive a performance score and feedback based on your responses.


# Future Improvements

Enhance keyword extraction with advanced NLP techniques (e.g., BERT-based models).
Integrate a web-based UI for a more user-friendly experience.
Add multi-language support for job descriptions and resumes.
Improve question generation with domain-specific customization.


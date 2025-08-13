<h1 align="center">🤖 JobSyncPro: Ai-Powered Recruitment Suite</h1>
<p align="center"><i>Optimize Resumes, Simulate Interviews — Your AI-Powered Career Assistant</i></p>
<p align="center">
  <img src="https://img.shields.io/badge/status-complete-green" alt="Project Status Badge"/>
<!--   <img src="https://img.shields.io/badge/tech-stack-React%20%7C%20Next.js%20%7C%20Python-blue" alt="Tech Stack Badge"/> -->
  <img src="https://img.shields.io/badge/API-Gemini%20%7C%20GenAI-lightgrey" alt="API Badge"/>
</p>
<h3 align="center"><i>Built with the tools and technologies:</i></h3>
<p align="center">
  <img src="https://img.shields.io/badge/-Flask-3776AB?logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/-SpaCy-green?logo=spaCy" alt="SpaCy"/>
  <img src="https://img.shields.io/badge/-Gemini_API-black?logo=google" alt="Gemini API"/>
  <img src="https://img.shields.io/badge/-CSS-purple?logo=CSS" alt="CSS"/>
  <img src="https://img.shields.io/badge/-JavaScript-000000?logo=javascript" alt="JavaScript"/>
  <img src="https://img.shields.io/badge/-HTML-61DAFB?logo=HTML&logoColor=black" alt="HTML"/>
  <img src="https://img.shields.io/badge/-Firebase-61RAFB?logo=firebase&logoColor=black" alt="Firebase"/>
</p>

---
## 📑 Table of Contents
* [Overview](#overview)
* [Features](#features)
  * [Resume ATS Scoring](#resume-ats-scoring)
  * [AI Interview System](#ai-interview-system)
* [Tech Stack](#tech-stack)
* [Installation](#installation)
* [Usage](#usage)
* [Future Improvements](#future-improvements)
* [Development Progress](#development-progress)
---
## 📌 Overview
**JobSyncPro** is an AI-powered platform with two main modules:
- 🎯 `Resume ATS Scoring System:` This module acts as your intelligent resume gatekeeper. It leverages advanced NLP techniques to extract crucial keywords from job descriptions. Its core strength lies in its `hybrid matching approach: combining Google Gemini's sophisticated semantic understanding (contributing 70% to the score) to grasp contextual relevance and infer skills, with traditional TF-IDF-based keyword similarity (contributing 30%) `for a robust statistical measure of direct term overlap. This ensures a comprehensive and fair assessment of how well a resume truly aligns with a job's requirements. Beyond just a score, it provides actionable improvement suggestions via Gemini.
- 🧠 `AI-Powered Interview System:` Following resume screening, this module takes over. It uses Generative AI to parse candidate resumes, intelligently generating personalized interview questions tailored to their unique experience. It then assesses candidate responses and delivers immediate, constructive feedback, preparing candidates and providing recruiters with deeper insights.
## ✨ Features
### ✅ Resume ATS Scoring
* **🔍 Keyword Extraction**: Uses SpaCy to identify and extract relevant terms from job descriptions.
* **📊 Resume Matching**: Compares resumes against job descriptions to compute an ATS score.
* **🧠 Smart Suggestions**: Uses Gemini API to offer actionable tips to boost resume-job fit.
### 🎤 AI Interview System

* **📂 Resume Parsing**: GenAI detects and organizes key resume sections (projects, skills, experience).
* **❓ Question Generation**: Personalized interview questions created using parsed data.
* **📝 Answer Evaluation**: Evaluates user responses and scores performance with feedback.
---
## ⚙️ Tech Stack

* **Backend**: Python, SpaCy, Gemini API, GenAI
* **Frontend**: React, Next.js *(in progress on [this repository](https://github.com/punitkumar4871/Resume_interview_matcher))*

---
## 📦 Installation
```bash
# 1. Clone the repository
git clone https://github.com/yashvisharma1204/JobSyncPro.git
# 2. Navigate to the directory
cd JobSyncPro
# 3. Install Python dependencies
pip install -r requirements.txt
# 4. Install SpaCy model
python -m spacy download en_core_web_sm
```
### 🔐 API Keys
* Create a `.env` file and add:
  ```
  GEMINI_API_KEY=your_gemini_key_here
  ```
* Configure any additional credentials required for GenAI functionality.
---
## 💡 Usage
### 🧾 ATS Scoring
* Upload or input a **job description** and **resume** (text/PDF).
* Get an ATS compatibility score and improvement suggestions.

### 🎙️ Interview System
* Answer AI-generated questions via CLI or Web UI.
* Receive scores and improvement feedback automatically.
---

## 🔮 Future Improvements
* Integrate BERT or transformer-based models for smarter keyword extraction.
* Add an intuitive web-based UI with dashboard.
* Enable multi-language support for global job seekers.
* Improve domain-specific interview question customization.

---
## 🚧 Development Progress
The frontend development using **React** and **Next.js** is ongoing at:
🔗 [https://github.com/punitkumar4871/Resume\_interview\_matcher](https://github.com/punitkumar4871/Resume_interview_matcher)
Stay tuned for exciting updates!

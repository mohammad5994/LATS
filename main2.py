from resume_parsing import resume_parser, resume_parser_openai
from job_desc_parsing import extract_jd_components_openai
from scoring_module import scoring_openai
import streamlit as st
import os


st.set_page_config(layout="wide")

st.title("Resume Optimizer")

col1, col2, col3 = st.columns(3, border=True)

# Inputs
uploaded_file = col1.file_uploader("Upload your pdf resume file", type=["pdf"])
job_desc = col1.text_area("Paste job description here...")
job_id = col1.text_input("Enter a job id")

# Initialize session state flags
if "extracted" not in st.session_state:
    st.session_state.extracted = False
if "tailored" not in st.session_state:
    st.session_state.tailored = False

# Handle extract button
if col1.button("Extract Resume Data", key="bt-11"):
    if uploaded_file is None or not job_desc.strip() or not job_id.strip():
        col1.warning("Please upload your resume, job description and job ID")
    else:
        st.session_state.extracted = True  # Set flag

# Show extract results if extracted
if st.session_state.extracted:
    if not os.path.exists(f"./data/{job_id}"):
        os.makedirs(f"./data/{job_id}")
    user_resume = resume_parser_openai(uploaded_file, job_id=job_id)

    col2.text_input("Job title", user_resume["title"])
    col2.text_area("Profile", user_resume["bio"])
    col2.text_area("Skills", " | ".join(user_resume["skills"]))
    col2.text_area("Professional Experiences", user_resume["work_experience"])
    col2.text_area("Education", user_resume["education"])
else:
    col2.warning("Extract Resume not clicked yet")

# Handle tailoring button
if col2.button("Start Tailoring Resume", key="bt-12"):
    if uploaded_file is None or not job_desc.strip() or not job_id.strip():
        col2.warning("Please upload your resume, job description and job ID")
    else:
        st.session_state.tailored = True  # Set flag

# Show tailored resume content
if st.session_state.tailored:
    extract_jd_components_openai(job_desc, job_id)
    (profile, responsibilities, tech_skills,
     domain_skills, soft_skills) = scoring_openai(job_id)
    col3.text_area("Profile", value=profile)
    col3.text_area("Professional Experience", value=responsibilities)
    col3.text_area("Technical-Skills", tech_skills)
    col3.text_area("Domain-Specific-Skills", domain_skills)
    col3.text_area("Soft-Skills", soft_skills)
else:
    col3.warning("Start Tailoring not clicked yet")

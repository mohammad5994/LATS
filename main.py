from resume_parsing import resume_parser
from job_desc_parsing import extract_jd_components_openai
from scoring_module import scoring_openai
import os
import streamlit as st

if not os.path.exists("./data"):
    os.makedirs("./data")

st.title("Resume Optimizer")

tab1, tab2, tab3 = st.tabs(["Initial Info", "Resume Info", "Optimization Suggestions"])
extract = False
optimise = False

with tab1:
    uploaded_file = st.file_uploader("Upload your pdf resume file", type=["pdf"])
    job_desc = st.text_area("Paste job description here...")
    job_id = st.text_input("Enter a job id")
    extract = st.button("Extract Resume Data", key=1)
with tab2:
    if extract:
        if uploaded_file is None or not job_desc.strip() or not job_id.strip():
            st.warning("Please upload your resume, job description and job ID")
            optimise = st.button("Start Tailoring Resume", disabled=True, key=2)
        else:
            if not os.path.exists(f"./data/{job_id}"):
                os.makedirs(f"./data/{job_id}")
            user_resume = resume_parser(uploaded_file, job_id=job_id)

            st.text_input("Job title", user_resume["title"])
            st.text_area("Profile", user_resume["bio"])
            st.text_area("Skills", " | ".join(user_resume["skills"]))
            st.text_area("Professional Experiences", user_resume["work_experience"])
            st.text_area("Education", user_resume["education"])
            optimise = st.button("Start Tailoring Resume", key=2)

with tab3:
    if optimise:
        if uploaded_file is None or not job_desc.strip() or not job_id.strip():
            st.warning("Please upload your resume, job description and job ID")
        else:
            extract_jd_components_openai(job_desc, job_id)
            (profile, responsibilities, tech_skills,
             domain_skills, soft_skills) = scoring_openai(job_id)
            st.text_area("Profile", value=profile)
            st.text_area("Professional Experience", value=responsibilities)
            st.text_area("Technical-Skills", tech_skills)
            st.text_area("Domain-Specific-Skills", domain_skills)
            st.text_area("Soft-Skills", soft_skills)

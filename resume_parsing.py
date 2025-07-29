import pdfplumber as pl
import string
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
from ollama import chat
from ollama import ChatResponse
import pickle
from openai import OpenAI
from dotenv import load_dotenv
import os

nltk.download()

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "llama3.2:3b"


def get_pdf_text(path, job_id):
    pdf = pl.open(path)

    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    with open(f"./data/{job_id}/resume_no_cleaning.txt", "a") as f:
        f.write(pdf_text)

    return pdf_text


def clean_text(text, stem=None):
    nlp = spacy.load('en_core_web_sm')

    cleaned_text = ""

    text = text.lower()

    text = re.sub(r'\n', '', text)

    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english")
    useless_words = useless_words + ["hi", "im"]

    filtered_text = [word for word in text if not word in useless_words]

    filtered_text_spacy = nlp(' '.join(filtered_text))
    stemmed_text_spacy = [y.lemma_ for y in filtered_text_spacy]
    cleaned_text_spacy = " ".join(stemmed_text_spacy)
    with open("./data/resume_cleaned_spacy.txt", "a") as f:
        f.write(cleaned_text_spacy)

    stemmer = PorterStemmer()
    stemmed_text_porter = [stemmer.stem(y) for y in filtered_text]
    cleaned_text_porter = " ".join(stemmed_text_porter)
    with open("./data/resume_cleaned_porter.txt", "a") as f:
        f.write(cleaned_text_porter)

    lem = WordNetLemmatizer()
    stemmed_text_wordnet = [lem.lemmatize(y) for y in filtered_text]
    cleaned_text_wordnet = " ".join(stemmed_text_wordnet)
    with open("./data/resume_cleaned_wordnet.txt", "a") as f:
        f.write(cleaned_text_wordnet)

    cleaned_text_pure = " ".join(filtered_text)
    with open("./data/resume_cleaned_pure.txt", "a") as f:
        f.write(cleaned_text_pure)


def extract_phone_number(resume_text):
    PHONE_REG = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
    phone = re.findall(PHONE_REG, resume_text)

    if phone:
        number = ''.join(phone[0])

        if resume_text.find(number) >= 0 and len(number) < 16:
            return number
    return "Not Found."


def extract_email(resume_text):
    EMAIL_REG = re.compile(r'[a-zA-Z0-9+_.-]+@[a-zA-Z0-9+_.-]+\.[a-zA-Z]{2,}')
    email = re.findall(EMAIL_REG, resume_text)

    if email:
        return email[0]
    return "Not Found."


def extract_linkedin(resume_text):
    LINKEDIN_REG = re.compile(r'[a-zA-Z0-9+_.-/:]+linkedin.com[^\s]+')
    linkedin = re.findall(LINKEDIN_REG, resume_text)

    if linkedin:
        return linkedin[0]
    return "Not Found."


def extract_github(resume_text):
    GITHUB_REG = re.compile(r'[a-zA-Z0-9+_.-/:]+github.com[^\s]+')
    git = re.findall(GITHUB_REG, resume_text)

    if git:
        return git[0]
    return "Not Found."


def extract_skills_section(resume_text):
    pattern = r'Skills\s*\n(.*?)(?=\n[A-Z][a-z]+\s*\n|\Z)'

    match = re.search(pattern, resume_text, re.DOTALL | re.IGNORECASE)

    if match:
        skills_content = match.group(1).strip()
        return skills_content
    else:
        return "Not Found."


def extract_skills(resume_text):
    skill_section = extract_skills_section(resume_text)
    response: ChatResponse = chat(model=MODEL_NAME, messages=[
        {
            'role': 'user',
            'content': f"""You are an AI assistant that extracts individual skill names from professional resumes, CVs, or descriptions of technical capabilities. Given a text describing someone's skills—whether they are in plain text, bullet lists, or separated by characters like `|`, `-`, `:`, or commas—your job is to identify and return a clean list of the **individual skill names**.
                            Return only the skills, no explanation.

                            Follow these instructions strictly:
                            1. Extract only **skill-related terms**, whether technical (like "Python", "SQL") or domain-specific (like "Applied AI", "IT Consulting").
                            2. **Return only the skill names**, one per line. **Do not include any explanation, introductory sentence, or labels.**
                            3. Do **not** include category headers like "Technical Skills" or "Domain-Specific Skills" or "Programming Language" or "Soft-Skills".
                            4. Normalize spaces (strip leading/trailing whitespace).
                            5. Do **not** include duplicates.
                            6. Keep multi-word skills intact (e.g., "Machine Learning", "Web Scraping").
                            
                            ---

                            Text:
                            {skill_section}
                            
                            ---
                            
                            Output (no intro, no numbering, no bullets, just skill names):

""",
        },
    ])
    extracted_skills = str(response['message']['content'])

    skills_list = [skill.strip() for skill in extracted_skills.splitlines() if skill.strip()]
    return skills_list


def extract_skills_openai(resume_text):
    skill_section = extract_skills_section(resume_text)
    openai = OpenAI(api_key=openai_api_key)
    response = openai.chat.completions.create(model="gpt-4.1-mini", messages=[
        {
            "role": "system",
            "content": f"""
                    You are a helpful resume analyzer that helps to extract specific part of resume. 
                    """
        },
        {
            'role': 'user',
            'content': f"""You are an AI assistant that extracts individual skill names from professional resumes, CVs, or descriptions of technical capabilities. Given a text describing someone's skills—whether they are in plain text, bullet lists, or separated by characters like `|`, `-`, `:`, or commas—your job is to identify and return a clean list of the **individual skill names**.
                            Return only the skills, no explanation.

                            Follow these instructions strictly:
                            1. Extract only **skill-related terms**, whether technical (like "Python", "SQL") or domain-specific (like "Applied AI", "IT Consulting").
                            2. **Return only the skill names**, one per line. **Do not include any explanation, introductory sentence, or labels.**
                            3. Do **not** include category headers like "Technical Skills" or "Domain-Specific Skills" or "Programming Language" or "Soft-Skills".
                            4. Normalize spaces (strip leading/trailing whitespace).
                            5. Do **not** include duplicates.
                            6. Keep multi-word skills intact (e.g., "Machine Learning", "Web Scraping").

                            ---

                            Text:
                            {skill_section}

                            ---

                            Output (no intro, no numbering, no bullets, just skill names):

""",
        },
    ])
    extracted_skills = str(response.choices[0].message.content)

    skills_list = [skill.strip() for skill in extracted_skills.splitlines() if skill.strip()]
    return skills_list


def extract_work_experience(resume_text):
    response: ChatResponse = chat(model=MODEL_NAME, messages=[
        {
            "role": "user",
            "content": f"""You are a resume analysis assistant. Given the text content of a resume, extract the full "Work Experience" section, including job titles, company names, employment dates, and any bullet points or descriptions associated with each position.

                ### Instructions:
                - Identify the section based on common headers like "Experience", "Work Experience", "Professional Experience", "Employment History", etc.
                - Include **all roles listed** in that section.
                - Do not include unrelated sections (like education, skills, or projects).
                - Maintain the formatting if possible (e.g., line breaks between roles).
                - If you do not find this section, just return Not found.
                
                ### Resume Text:
                {resume_text}
                
                ### Output:
                Return only the Work Experience section in clean, readable format (no intro, no numbering, no explanation).
            """
        },
    ])

    return str(response["message"]["content"])


def extract_work_experience_openai(resume_text):
    openai = OpenAI(api_key=openai_api_key)
    response = openai.chat.completions.create(model="gpt-4.1-mini", messages=[
        {
            "role": "system",
            "content": f"""
                You are a helpful resume analyzer that helps to extract specific part of resume. 
                """
        },
        {
            "role": "user",
            "content": f"""You are a resume analysis assistant. Given the text content of a resume, extract the full "Work Experience" section, including job titles, company names, employment dates, and any bullet points or descriptions associated with each position.

                ### Instructions:
                - Identify the section based on common headers like "Experience", "Work Experience", "Professional Experience", "Employment History", etc.
                - Include **all roles listed** in that section.
                - Do not include unrelated sections (like education, skills, or projects).
                - Maintain the formatting if possible (e.g., line breaks between roles).
                - If you do not find this section, just return Not found.

                ### Resume Text:
                {resume_text}

                ### Output:
                Return only the Work Experience section in clean, readable format (no intro, no numbering, no explanation).
            """
        },
    ])

    return str(response.choices[0].message.content)


def extract_education(resume_text):
    response: ChatResponse = chat(model=MODEL_NAME, messages=[
        {
            "role": "user",
            "content": f"""
            You are a resume analysis assistant. Given the text content of a resume, extract the full "Education" section, including degree titles(MSc, BSc, etc), program name(ICT, software engineering, biomedical engineering, etc), university names, education dates, and any bullet points or descriptions associated with each degree program.

                ### Instructions:
                - Identify the section based on common headers like "Education", "Academic Background", etc.
                - Include **all educations listed** in that section.
                - Do not include unrelated sections (like work experience, skills, or projects).
                - Maintain the formatting if possible (e.g., line breaks between educations).
                - If you do not find this section, just return Not found.
                
                ### Resume Text:
                {resume_text}
                
                ### Output:
                Return only the Education section in clean, readable format (no intro, no numbering, no explanation).
            """

        }
    ])

    return response["message"]["content"]


def extract_education_openai(resume_text):
    openai = OpenAI(api_key=openai_api_key)
    response = openai.chat.completions.create(model="gpt-4.1-mini", messages=[
        {
            "role": "system",
            "content": f"""
            You are a helpful resume analyzer that helps to extract specific part of resume. 
            """
        },
        {
            "role": "user",
            "content": f"""
            You are a resume analysis assistant. Given the text content of a resume, extract the full "Education" section, including degree titles(MSc, BSc, etc), program name(ICT, software engineering, biomedical engineering, etc), university names, education dates, and any bullet points or descriptions associated with each degree program.

                ### Instructions:
                - Identify the section based on common headers like "Education", "Academic Background", etc.
                - Include **all educations listed** in that section.
                - Do not include unrelated sections (like work experience, skills, or projects).
                - Maintain the formatting if possible (e.g., line breaks between educations).
                - If you do not find this section, just return Not found.

                ### Resume Text:
                {resume_text}

                ### Output:
                Return only the Education section in clean, readable format (no intro, no numbering, no explanation).
            """

        }
    ])

    return response.choices[0].message.content


def extract_projects(resume_text):
    response: ChatResponse = chat(model=MODEL_NAME, messages=[
        {
            "role": "user",
            "content": f"""
            You are a resume analysis assistant. Given the text content of a resume, extract the full "Projects" section, including project titles, project dates, and any bullet points or descriptions associated with each project.

                ### Instructions:
                - Identify the section based on common headers like "Projects".
                - Include **all projects listed** in that section.
                - Do not include unrelated sections (like work experience, skills, education, or projects).
                - Maintain the formatting if possible (e.g., line breaks between projects).
                - If you do not find this section, just return Not found.
                
                ### Resume Text:
                {resume_text}
                
                ### Output:
                Return only the Projects section in clean, readable format (no intro, no numbering, no explanation).
            """
        }
    ])

    return response["message"]["content"]


def extract_projects_openai(resume_text):
    openai = OpenAI(api_key=openai_api_key)
    response = openai.chat.completions.create(model="gpt-4.1-mini", messages=[
        {
            "role": "system",
            "content": f"""
                            You are a helpful resume analyzer that helps to extract specific part of resume. 
                            """
        },
        {
            "role": "user",
            "content": f"""
            You are a resume analysis assistant. Given the text content of a resume, extract the full "Projects" section, including project titles, project dates, and any bullet points or descriptions associated with each project.

                ### Instructions:
                - Identify the section based on common headers like "Projects".
                - Include **all projects listed** in that section.
                - Do not include unrelated sections (like work experience, skills, education, or projects).
                - Maintain the formatting if possible (e.g., line breaks between projects).
                - If you do not find this section, just return Not found.

                ### Resume Text:
                {resume_text}

                ### Output:
                Return only the Projects section in clean, readable format (no intro, no numbering, no explanation).
            """
        }
    ])

    return response.choices[0].message.content


def extract_certificates(resume_text):
    response: ChatResponse = chat(model=MODEL_NAME, messages=[
        {
            "role": "user",
            "content": f"""
            You are a resume analysis assistant. Given the text content of a resume, extract the full "Certificates" section, including Certificate titles, Certificate issuing dates, and any bullet points or descriptions associated with each certificate.

                ### Instructions:
                - Identify the section which exactly named as "Certificates".
                - Include **all Certificates listed** in that section.
                - Do not include unrelated sections (like work experience, skills, education, or projects).
                - Maintain the formatting if possible (e.g., line breaks between Certificates).
                - If you do not find this section, just return Not found.

                ### Resume Text:
                {resume_text}

                ### Output:
                Return only the Certificates section in clean, readable format (no intro, no numbering, no explanation).
            """
        }
    ])

    return response["message"]["content"]


def extract_certificates_openai(resume_text):
    openai = OpenAI(api_key=openai_api_key)
    response = openai.chat.completions.create(model="gpt-4.1-mini", messages=[
        {
            "role": "system",
            "content": f"""
                        You are a helpful resume analyzer that helps to extract specific part of resume. 
                        """
        },
        {
            "role": "user",
            "content": f"""
            You are a resume analysis assistant. Given the text content of a resume, extract the full "Certificates" section, including Certificate titles, Certificate issuing dates, and any bullet points or descriptions associated with each certificate.

                ### Instructions:
                - Identify the section which exactly named as "Certificates".
                - Include **all Certificates listed** in that section.
                - Do not include unrelated sections (like work experience, skills, education, or projects).
                - Maintain the formatting if possible (e.g., line breaks between Certificates).
                - If you do not find this section, just return Not found.

                ### Resume Text:
                {resume_text}

                ### Output:
                Return only the Certificates section in clean, readable format (no intro, no numbering, no explanation).
            """
        }
    ])

    return response.choices[0].message.content


def extract_publications(resume_text):
    response: ChatResponse = chat(model=MODEL_NAME, messages=[
        {
            "role": "user",
            "content": f"""
            You are a resume analysis assistant. Given the text content of a resume, extract the full "Publication" section, including publication titles, publication submitted or accepted dates, and any bullet points or descriptions associated with each publication.

                ### Instructions:
                - Identify the section which exactly named as "Publication".
                - Include **all publication listed** in that section.
                - Do not include unrelated sections (like work experience, skills, education, or projects).
                - Maintain the formatting if possible (e.g., line breaks between publications).
                - If you do not find this section, just return Not found.

                ### Resume Text:
                {resume_text}

                ### Output:
                Return only the publication section in clean, readable format (no intro, no numbering, no explanation).
            """
        }
    ])

    return response["message"]["content"]


def extract_publications_openai(resume_text):
    openai = OpenAI(api_key=openai_api_key)
    response = openai.chat.completions.create(model="gpt-4.1-mini", messages=[
        {
            "role": "system",
            "content": f"""
                    You are a helpful resume analyzer that helps to extract specific part of resume. 
                    """
        },
        {
            "role": "user",
            "content": f"""
            You are a resume analysis assistant. Given the text content of a resume, extract the full "Publication" section, including publication titles, publication submitted or accepted dates, and any bullet points or descriptions associated with each publication.

                ### Instructions:
                - Identify the section which exactly named as "Publication".
                - Include **all publication listed** in that section.
                - Do not include unrelated sections (like work experience, skills, education, or projects).
                - Maintain the formatting if possible (e.g., line breaks between publications).
                - If you do not find this section, just return Not found.

                ### Resume Text:
                {resume_text}

                ### Output:
                Return only the publication section in clean, readable format (no intro, no numbering, no explanation).
            """
        }
    ])

    return response.choices[0].message.content


def extract_languages(resume_text):
    response: ChatResponse = chat(model=MODEL_NAME, messages=[
        {
            "role": "user",
            "content": f"""
            You are a resume analysis assistant. Given the text content of a resume, extract the full "Language" section, including languages name and any bullet points or descriptions associated with each language.

                ### Instructions:
                - Identify the section based on common headers like "Language".
                - Include **all languages listed** in that section.
                - Do not include unrelated sections (like work experience, skills, education, or projects).
                - Maintain the formatting if possible (e.g., line breaks between educations).
                - If you do not find this section, just return Not found.

                ### Resume Text:
                {resume_text}

                ### Output:
                Return only the languages section in clean, readable format (no intro, no numbering, no explanation).
            """
        }
    ])

    return response["message"]["content"]


def extract_languages_openai(resume_text):
    openai = OpenAI(api_key=openai_api_key)
    response = openai.chat.completions.create(model="gpt-4.1-mini", messages=[
        {
            "role": "system",
            "content": f"""
                You are a helpful resume analyzer that helps to extract specific part of resume. 
                """
        },
        {
            "role": "user",
            "content": f"""
            You are a resume analysis assistant. Given the text content of a resume, extract the full "Language" section, including languages name and any bullet points or descriptions associated with each language.

                ### Instructions:
                - Identify the section based on common headers like "Language".
                - Include **all languages listed** in that section.
                - Do not include unrelated sections (like work experience, skills, education, or projects).
                - Maintain the formatting if possible (e.g., line breaks between educations).
                - If you do not find this section, just return Not found.

                ### Resume Text:
                {resume_text}

                ### Output:
                Return only the languages section in clean, readable format (no intro, no numbering, no explanation).
            """
        }
    ])

    return response.choices[0].message.content


def extract_title(resume_text):
    response: ChatResponse = chat(model=MODEL_NAME, messages=[
        {
            "role": "user",
            "content": f"""
            You are a resume analysis assistant. Given the text content of a resume, extract the the role that the applicant wants to apply for it. It usually should be at the tope of resume around the Name of the applicant.

                ### Instructions:
                - Identify the title around the personal information of the applicant, the title is two or three words not a sentence.
                - Do not include unrelated information (like work experience, skills, education, or projects).
                - Maintain the formatting if possible.
                - If you do not find the title, just return Not found.

                ### Resume Text:
                {resume_text}

                ### Output:
                Return only the title clean, readable format (no intro, no numbering, no explanation).
            """
        }
    ])

    return response["message"]["content"]


def extract_title_openai(resume_text):
    openai = OpenAI(api_key=openai_api_key)
    response = openai.chat.completions.create(model="gpt-4.1-mini", messages=[
        {
            "role": "system",
            "content": f"""
            You are a helpful resume analyzer that helps to extract specific part of resume. 
            """
        },
        {
            "role": "user",
            "content": f"""
            You are a resume analysis assistant. Given the text content of a resume, extract the the role that the applicant wants to apply for it. It usually should be at the tope of resume around the Name of the applicant.

                ### Instructions:
                - Identify the title around the personal information of the applicant, the title is two or three words not a sentence.
                - Do not include unrelated information (like work experience, skills, education, or projects).
                - Maintain the formatting if possible.
                - If you do not find the title, just return Not found.

                ### Resume Text:
                {resume_text}

                ### Output:
                Return only the title clean, readable format (no intro, no numbering, no explanation).
            """
        }
    ])

    return response.choices[0].message.content


def extract_bio(resume_text):
    response: ChatResponse = chat(model=MODEL_NAME, messages=[
        {
            "role": "user",
            "content": f"""
                You are a resume analysis assistant. Given the text content of a resume, extract the the Profile section of the resume that is summary of the applicant’s background, qualifications, or professional goals. It usually should be at the top of resume.

                    Be aware that the Profile section can have different headings, such as:

                    Profile
                    
                    Summary
                    
                    CV Summary
                    
                    Professional Summary
                    
                    Bio
                    
                    About Me
                    
                    Personal Statement
                    
                    Executive Summary
                    
                    When extracting the section:
                    
                    Detect common section headings that indicate a profile/summary.
                    
                    Extract only the content under the detected heading, stopping when the next major section begins (e.g., "Work Experience", "Education", "Skills").
                    
                    If no clear heading is found, you may infer the section if the first paragraph(s) at the beginning of the CV have a summary-like tone.
                    
                    Do not include any section headers or unrelated text.
                    
                    Return only the content of the profile section.
                    
                    return "Not Found" if you do not find any related section in the resume.
                    


                    ### Resume Text:
                    {resume_text}

                    ### Output:
                    Return only the profile section in a clean, readable format (no intro, no numbering, no explanation).
                """
        }
    ])

    return response["message"]["content"]


def extract_bio_openai(resume_text):
    prompts = [
        {
            "role": "system",
            "content": """
            You are a helpful resume analyzer that helps to extract specific part of resume. 
            """
        },
        {
            "role": "user",
            "content": f"""
                You are a resume analysis assistant. Given the text content of a resume, extract the the Profile section of the resume that is summary of the applicant’s background, qualifications, or professional goals. It usually should be at the top of resume.

                    Be aware that the Profile section can have different headings, such as:

                    Profile

                    Summary

                    CV Summary

                    Professional Summary

                    Bio

                    About Me

                    Personal Statement

                    Executive Summary

                    When extracting the section:

                    Detect common section headings that indicate a profile/summary.

                    Extract only the content under the detected heading, stopping when the next major section begins (e.g., "Work Experience", "Education", "Skills").

                    If no clear heading is found, you may infer the section if the first paragraph(s) at the beginning of the CV have a summary-like tone.

                    Do not include any section headers or unrelated text.

                    Return only the content of the profile section.

                    return "Not Found" if you do not find any related section in the resume.



                    ### Resume Text:
                    {resume_text}

                    ### Output:
                    Return only the profile section in a clean, readable format (no intro, no numbering, no explanation).
                """
        }
    ]
    openai = OpenAI(api_key=openai_api_key)
    response = openai.chat.completions.create(model="gpt-4.1-mini", messages=prompts)

    return response.choices[0].message.content


def resume_parser(path, job_id):
    print(f"Start parsing resume...")
    user_resume_items = {}

    pdf_text = get_pdf_text(path)
    #print(pdf_text)

    title = extract_title(pdf_text)
    user_resume_items["title"] = title

    phone_number = extract_phone_number(pdf_text)
    user_resume_items["phone"] = phone_number

    email = extract_email(pdf_text)
    user_resume_items["email"] = email

    linkedin = extract_linkedin(pdf_text)
    user_resume_items["linkedin"] = linkedin

    github = extract_github(pdf_text)
    user_resume_items["github"] = github

    bio = extract_bio(pdf_text)
    user_resume_items["bio"] = bio

    skills = extract_skills(pdf_text)
    user_resume_items["skills"] = skills

    work_exp_section = extract_work_experience(pdf_text)
    user_resume_items["work_experience"] = work_exp_section

    education = extract_education(pdf_text)
    user_resume_items["education"] = education

    projects = extract_projects(pdf_text)
    user_resume_items["projects"] = projects

    publications = extract_publications(pdf_text)
    user_resume_items["publications"] = publications

    certificates = extract_certificates(pdf_text)
    user_resume_items["certificates"] = certificates

    languages = extract_languages(pdf_text)
    user_resume_items["languages"] = languages

    print(user_resume_items)
    with open(f'./data/{job_id}/resume_json.p', 'wb') as fp:
        pickle.dump(user_resume_items, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return user_resume_items


def resume_parser_openai(path, job_id):
    print(f"Start parsing resume...")
    user_resume_items = {}

    pdf_text = get_pdf_text(path, job_id)
    #print(pdf_text)

    title = extract_title_openai(pdf_text)
    user_resume_items["title"] = title

    phone_number = extract_phone_number(pdf_text)
    user_resume_items["phone"] = phone_number

    email = extract_email(pdf_text)
    user_resume_items["email"] = email

    linkedin = extract_linkedin(pdf_text)
    user_resume_items["linkedin"] = linkedin

    github = extract_github(pdf_text)
    user_resume_items["github"] = github

    bio = extract_bio_openai(pdf_text)
    user_resume_items["bio"] = bio

    skills = extract_skills_openai(pdf_text)
    user_resume_items["skills"] = skills

    work_exp_section = extract_work_experience_openai(pdf_text)
    user_resume_items["work_experience"] = work_exp_section

    education = extract_education_openai(pdf_text)
    user_resume_items["education"] = education

    projects = extract_projects_openai(pdf_text)
    user_resume_items["projects"] = projects

    publications = extract_publications_openai(pdf_text)
    user_resume_items["publications"] = publications

    certificates = extract_certificates_openai(pdf_text)
    user_resume_items["certificates"] = certificates

    languages = extract_languages_openai(pdf_text)
    user_resume_items["languages"] = languages

    print(user_resume_items)
    with open(f'./data/{job_id}/resume_json.p', 'wb') as fp:
        pickle.dump(user_resume_items, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return user_resume_items


#pdf_text = get_pdf_text("./data/resume.pdf")
#print(extract_bio(pdf_text))

"""user_resume_items = {}
title = extract_title(pdf_text)
user_resume_items["title"] = title

phone_number = extract_phone_number(pdf_text)
user_resume_items["phone"] = phone_number

email = extract_email(pdf_text)
user_resume_items["email"] = email

linkedin = extract_linkedin(pdf_text)
user_resume_items["linkedin"] = linkedin

git = extract_github(pdf_text)
user_resume_items["github"] = git

print(user_resume_items)"""

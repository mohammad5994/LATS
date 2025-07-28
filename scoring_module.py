from ollama import chat, ChatResponse
from openai import OpenAI
import json
import pickle
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
import string
from collections import defaultdict
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "llama3.2:3b"


def text_splitter(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
    )
    chunks = splitter.split_text(text)
    return chunks


def text_cleaning(text):
    cleaned_text = ""

    text = text.lower()

    text = re.sub(r'\n', '', text)

    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english")
    useless_words = useless_words + ["hi", "im"]

    #cleaned_text = text
    cleaned_text = [word for word in text if not word in useless_words]

    return cleaned_text


def get_CV_responsibilities_chunks(job_id):
    with open('./data/resume_json.p', 'rb') as fp:
        resume = pickle.load(fp)

    text = resume["work_experience"].replace("•", "")
    #text = text_cleaning(text)
    #chunks = text_splitter(" ".join(text))
    return text

def get_CV_bio(job_id):
    with open('./data/resume_json.p', 'rb') as fp:
        resume = pickle.load(fp)

    text = resume["bio"].replace("•", "")
    #text = text_cleaning(text)
    #chunks = text_splitter(" ".join(text))
    return text


def get_JD_responsibilities_chunks(job_id):
    with open('./data/jd_llama.txt', 'r') as f:
        jd = json.load(f)

    #text = text_cleaning(" ".join(jd["responsibilities"]))
    #chunks = text_splitter(" ".join(text))
    return "\n".join(jd["responsibilities"])


def get_JD_skills(job_id):
    with open('./data/jd_gpt.txt', 'r') as f:
        jd = json.load(f)

    skills = jd["required_skills"] + jd["preferred_skills"] + jd["soft_skills"]
    print("Start Cleaning JD skills...")
    #print(f"len JD skills: {len(skills)}")
    cleaned_skills = []
    for i, skill in enumerate(skills):
        #print(f"cleaning: {skill}")
        cleaned = text_cleaning(skill)
        if len(cleaned) > 1:
            cleaned_skills.append(" ".join(cleaned))
        else:
            cleaned_skills.append("".join(cleaned))
    return skills, cleaned_skills


def get_CV_skills(job_id):
    with open('./data/resume_json.p', 'rb') as fp:
        resume = pickle.load(fp)
    skills = []
    print("Start Cleaning cv skills...")
    for i, skill in enumerate(resume["skills"]):
        cleaned = text_cleaning(skill)
        if len(cleaned) > 1:
            skills.append(" ".join(cleaned))
        else:
            skills.append("".join(cleaned))
    return resume["skills"], skills


def compute_responsibilities_score(job_id):
    jd_chunks = get_JD_responsibilities_chunks(job_id)
    cv_chunks = get_CV_responsibilities_chunks(job_id)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    resume_vecs = model.encode(cv_chunks, normalize_embeddings=True)
    jd_vecs = model.encode(jd_chunks, normalize_embeddings=True)

    similarity_matrix = cosine_similarity(resume_vecs, jd_vecs)

    print(f"Similarity: {similarity_matrix}")


def compute_skills_score(job_id):
    cv_skills_org, cv_skills_cleaned = get_CV_skills(job_id)
    jd_skills_org, jd_skills_cleaned = get_JD_skills(job_id)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    resume_vecs = model.encode(cv_skills_cleaned, normalize_embeddings=True)
    jd_vecs = model.encode(jd_skills_cleaned, normalize_embeddings=True)

    similarity_matrix = cosine_similarity(resume_vecs, jd_vecs)

    print(f"Similarity: {similarity_matrix}")

    matched_skills = []
    for j, jd_skill in enumerate(jd_skills_org):
        best_cv_idx = np.argmax(similarity_matrix[:, j])
        best_similarity = similarity_matrix[best_cv_idx][j]
        if best_similarity > 0.50:
            matched_skills.append(
                {"job_skill": jd_skill,
                 "matched_cv_skill": cv_skills_org[best_cv_idx]})
        else:
            matched_skills.append(
                {"job_skill": jd_skill,
                 "matched_cv_skill": "Not Found"})

    with open(f"./data/{job_id}/matched_skills_embedding.txt", "w") as fp:
        fp.write(json.dumps(matched_skills))


def compute_skills_score_gpt(job_id):
    cv_skills_org, cv_skills_cleaned = get_CV_skills(job_id)
    jd_skills_org, jd_skills_cleaned = get_JD_skills(job_id)
    openai = OpenAI(api_key=openai_api_key)

    prompts = [
        {
            "role": "system",
            "content": """
            You are a helpful assistant that compares two lists of skills: one from a job description and the other from a user's CV. Your job is to semantically match each job description skill with the most relevant skill from the user's CV. If there is no good semantic match, return "Not Found".
                       """
        },
        {
            "role": "user",
            "content": """
            I have two lists of strings in Python:

- `job_skills`: A list of required skills extracted from a job description. These are phrases or sentences, e.g., ["Experience with cloud services like AWS", "Proficiency in Python", "Knowledge of Docker and containerization"].

- `cv_skills`: A list of skills from my CV, also as phrases or sentences, e.g., ["Skilled in Python and Flask", "Worked with Docker for deployment", "Familiar with Azure cloud"].

Your task is to semantically match each skill in `job_skills` with the most similar skill in `cv_skills`. If no relevant or similar skill is found, return `"Not Found"` for that entry.

Return the result as a list of dictionaries like this(no intro, no labeling, no explanation):
[
    {"job_skill": "<job_skill_1>", "matched_cv_skill": "<cv_skill_or_Not Found>"},
    ...
]

Requirements:

Use semantic similarity (not just keyword or string matching).

Match only one CV skill per job skill — the best one.

If the best match is not meaningfully related, return "Not Found".

job_skills = 
""" + json.dumps(jd_skills_org, indent=4) +
                       "cv_skills = " + json.dumps(cv_skills_org, indent=4)
        }
    ]
    response = openai.chat.completions.create(model="gpt-4o-mini", messages=prompts)
    with open(f"./data/{job_id}/matched_skills_gpt.txt", "w") as fp:
        fp.write(response.choices[0].message.content)


def compute_skills_score_llama(job_id):
    cv_skills_org, cv_skills_cleaned = get_CV_skills(job_id)
    jd_skills_org, jd_skills_cleaned = get_JD_skills(job_id)

    prompts = [
        {
            "role": "user",
            "content": """
            You are a helpful assistant that compares two lists of skills: one from a job description and the other from a user's CV. Your task is to semantically match each skill in `job_skills` with the most similar skill in `cv_skills`. If no relevant or similar skill is found, return `"Not Found"` for that entry.

Return the result as a list of dictionaries like this(no intro, no labeling, no explanation, no python code):
[
    {"job_skill": "<job_skill_1>", "matched_cv_skill": "<cv_skill_or_Not Found>"},
    ...
]

Requirements:

Use semantic similarity (not just keyword or string matching).

Match only one CV skill per job skill — the best one.

If the best match is not meaningfully related, return "Not Found".

Do not provide any python code in your response.

In the following you can find the list:

job_skills = 
""" + json.dumps(jd_skills_org, indent=4) +
                       "cv_skills = " + json.dumps(cv_skills_org, indent=4)
        }
    ]

    response: ChatResponse = chat(model=MODEL_NAME, messages=prompts)
    with open(f"./data/{job_id}/matched_skills_llama.txt", "w") as fp:
        fp.write(response["message"]["content"])


def compute_skills_score_qwen(job_id):
    cv_skills_org, cv_skills_cleaned = get_CV_skills(job_id)
    jd_skills_org, jd_skills_cleaned = get_JD_skills(job_id)

    prompts = [
        {
            "role": "user",
            "content": """
            You are a helpful assistant that compares two lists of skills and return a json: one list from a job description and the other list from a user's CV. Your task is to semantically match each skill in `job_skills` with the most similar skill in `cv_skills`. If no relevant or similar skill is found, return `"Not Found"` for that entry.

Return the result as a json like this:
[
    {"job_skill": "<job_skill_1>", "matched_cv_skill": "<cv_skill_or_Not Found>"},
    ...
]

Requirements:

Use semantic similarity (not just keyword or string matching).

Match only one CV skill per job skill — the best one.

If the best match is not meaningfully related, return "Not Found".

Do not provide any python code in your response.

Do not provide any explanation, intro, python code or labeling in the response, just return the json.

Do not include the <think>...</think> in the response.

In the following you can find the list:

job_skills = 
""" + json.dumps(jd_skills_org, indent=4) +
                       "\n cv_skills = " + json.dumps(cv_skills_org, indent=4)
        }
    ]

    response: ChatResponse = chat(model="qwen3:4b", messages=prompts)
    pattern = r'<think>.*?</think>'
    # Remove the think tags and their contents
    cleaned_text = re.sub(pattern, '', response["message"]["content"], flags=re.DOTALL)
    # Clean up any extra whitespace
    cleaned_text = cleaned_text.strip()
    with open(f"./data/{job_id}/matched_skills_qwen.txt", "w") as fp:
        fp.write(cleaned_text)


def skill_recommender_gpt(job_id):
    with open(f"./data/{job_id}/matched_skills_gpt.txt", "r") as fp:
        skills = json.load(fp)

    openai = OpenAI(api_key=openai_api_key)
    prompts = [
        {
            "role": "system",
            "content": """
                You are an expert AI assistant skilled in resume analysis and job skill classification. You will receive a list of dictionaries. Each dictionary contains two fields:
    
                "job_skill": a string describing a required job skill
                
                "matched_cv_skill": a string indicating the best-matched skill found in a candidate’s CV. If no match is found, this field contains "Not Found".
                
                Your task involves two steps for each dictionary item:
                
                Suggest a matched CV skill (for items where "matched_cv_skill" is "Not Found"). This suggestion should be a reasonable and specific skill a candidate might have that matches the given job requirement, keep the suggestion short and acuurate, only mention the exact skills that matches with the skill in the job description.
                
                Classify each skill into one of the following categories:
                
                "Technical-Skills": Programming languages, frameworks, tools, or technologies.
                
                "Domain-Specific Skills": Practical or applied knowledge in specific work domains (e.g., machine learning, data pipelines, business logic).
                
                "Soft-Skills": Interpersonal or personal abilities (e.g., communication, teamwork).
                
                You must output a new list of dictionaries, without any explanation, intro, labeling. Each dictionary must contain the original fields plus the following two:
                
                "suggested_cv_skill": your suggested skill (even if "matched_cv_skill" is not "Not Found", copy its value here).
                
                "category": the appropriate category for the skill.
                
                Maintain the original order. Here's the expected output format:
                
                {
                  "job_skill": "...",
                  "matched_cv_skill": "...",
                  "suggested_cv_skill": "...",
                  "category": "..."
                }
            """
        },
        {
            "role": "user",
            "content": """
            Use the following input list and generate the enhanced output accordingly:
            
            """ + json.dumps(skills)
        }
    ]

    response = openai.chat.completions.create(model="gpt-4o-mini", messages=prompts)
    with open(f"./data/{job_id}/enhanced_skills_gpt.txt", "w") as fp:
        fp.write(response.choices[0].message.content)

    return response.choices[0].message.content


def skill_recommender_qwen(job_id):
    with open(f"./data/{job_id}/matched_skills_qwen.txt", "r") as fp:
        skills = json.load(fp)

    prompts = [
        {
            "role": "user",
            "content": """
                You are an expert AI assistant skilled in resume analysis and job skill classification. You will receive a list of dictionaries. Each dictionary contains two fields:

                "job_skill": a string describing a required job skill

                "matched_cv_skill": a string indicating the best-matched skill found in a candidate’s CV. If no match is found, this field contains "Not Found".

                Your task involves two steps for each dictionary item:

                Suggest a matched CV skill (for items where "matched_cv_skill" is "Not Found"). This suggestion should be a reasonable and specific skill a candidate might have that matches the given job requirement.

                Classify each skill into one of the following categories:

                "Technical-Skills": Programming languages, frameworks, tools, or technologies.

                "Domain-Specific Skills": Practical or applied knowledge in specific work domains (e.g., machine learning, data pipelines, business logic).

                "Soft-Skills": Interpersonal or personal abilities (e.g., communication, teamwork).

                You must output a new list of dictionaries, without any explanation, intro, labeling. Each dictionary must contain the original fields plus the following two:

                "suggested_cv_skill": your suggested skill (even if "matched_cv_skill" is not "Not Found", copy its value here).

                "category": the appropriate category for the skill.

                Maintain the original order. Here's the expected output format:

                {
                  "job_skill": "...",
                  "matched_cv_skill": "...",
                  "suggested_cv_skill": "...",
                  "category": "..."
                }
                
                Use the following input list and generate the enhanced output accordingly:

            """ + json.dumps(skills)
        }
    ]

    response: ChatResponse = chat(model="qwen3:4b", messages=prompts)
    pattern = r'<think>.*?</think>'
    # Remove the think tags and their contents
    cleaned_text = re.sub(pattern, '', response["message"]["content"], flags=re.DOTALL)
    # Clean up any extra whitespace
    cleaned_text = cleaned_text.strip()
    with open(f"./data/{job_id}/enhanced_skills_qwen.txt", "w") as fp:
        fp.write(cleaned_text)


def responsibilities_recommender_gpt(job_id):
    jd_resp = get_JD_responsibilities_chunks(job_id)
    cv_resp = get_CV_responsibilities_chunks(job_id)
    with open(f"./data/{job_id}/enhanced_skills_gpt.txt", "r") as fp:
        skills = json.load(fp)
        suggested_skills = [
            item["suggested_cv_skill"]
            for item in skills
            if item["category"] in ["Domain-Specific Skills", "Technical-Skills"]
        ]

    prompts = [
        {
            "role": "system",
            "content": """
            You are a helpful and expert assistant skilled in resume optimization and job matching. Your task is to revise a job applicant’s professional experience section based on a target job description. You must intelligently align the candidate’s past responsibilities with the responsibilities and required skills from the job description, while preserving the structure and truthfulness of the resume.

            When rewriting, follow these principles:
            
            Keep the structure of the CV intact, including job title, company name, and dates of employment.
            
            Rewrite or reframe the responsibilities and achievements to better match the job description responsibilities. Include quantitative results in the professional experience section. The quantitative results must look real as much as possible.
                        
            Use the most relevant technical and domain-specific skills from the job description where appropriate.
            
            You may omit less relevant responsibilities or reword them for alignment, but do not fabricate experience.
            
            Keep the tone professional and concise, using action verbs and impact-oriented phrasing.
            
            You may group or reorder the bullet points if needed for clarity and relevance.
            
            The output must only be the professional experience section without any explanation, intro, labeling.
            """
        },
        {
            "role": "user",
            "content": f"""
            Here is the input data:

            1. Responsibilities from the job description:
            {jd_resp}
            \n
            2. Required technical and domain-specific skills:
            {suggested_skills}
            \n
            3. Professional Experience Section from the CV (to be rewritten):
            {cv_resp}
            \n
            Please rewrite the professional experience section to reflect the job description's responsibilities and preferred skills, maintaining factual accuracy and the original structure.
            """
        }
    ]
    openai = OpenAI(api_key=openai_api_key)
    response = openai.chat.completions.create(model="gpt-4o-mini", messages=prompts)
    with open(f"./data/{job_id}/professional_exp_gpt.txt", "w") as fp:
        fp.write(response.choices[0].message.content)

    return response.choices[0].message.content


def responsibilities_recommender_qwen(job_id):
    jd_resp = get_JD_responsibilities_chunks(job_id)
    cv_resp = get_CV_responsibilities_chunks(job_id)
    with open(f"./data/{job_id}/enhanced_skills_gpt.txt", "r") as fp:
        skills = json.load(fp)
        suggested_skills = [
            item["suggested_cv_skill"]
            for item in skills
            if item["category"] in ["Domain-Specific Skills", "Technical-Skills"]
        ]

    prompts = [
        {
            "role": "user",
            "content": f"""
            You are a helpful and expert assistant skilled in resume optimization and job matching. Your task is to revise a job applicant’s professional experience section based on a target job description. You must intelligently align the candidate’s past responsibilities with the responsibilities and required skills from the job description, while preserving the structure and truthfulness of the resume.

            When rewriting, follow these principles:

            Keep the structure of the CV intact, including job title, company name, and dates of employment.

            Rewrite or reframe the responsibilities and achievements to better match the job description responsibilities.
            
            Include quantitative results where applicable. The quantitative results must look real as much as possible.

            Use the most relevant technical and domain-specific skills from the job description where appropriate.

            You may omit less relevant responsibilities or reword them for alignment, but do not fabricate experience.

            Keep the tone professional and concise, using action verbs and impact-oriented phrasing.

            You may group or reorder the bullet points if needed for clarity and relevance.

            The output must only be the professional experience section without any explanation, intro, labeling.
            Here is the input data:

            1. Responsibilities from the job description:
            {jd_resp}
            \n
            2. Required technical and domain-specific skills:
            {suggested_skills}
            \n
            3. Professional Experience Section from the CV (to be rewritten):
            {cv_resp}
            \n
            Please rewrite the professional experience section to reflect the job description's responsibilities and preferred skills, maintaining factual accuracy and the original structure.
            """
        }
    ]
    response: ChatResponse = chat(model="qwen3:4b", messages=prompts)
    pattern = r'<think>.*?</think>'
    # Remove the think tags and their contents
    cleaned_text = re.sub(pattern, '', response["message"]["content"], flags=re.DOTALL)
    # Clean up any extra whitespace
    cleaned_text = cleaned_text.strip()
    with open(f"./data/{job_id}/professional_exp_qwen.txt", "w") as fp:
        fp.write(cleaned_text)


def cv_bio_recommender_qwen(job_id):
    cv_bio = get_CV_bio(job_id)
    jd_resp = get_JD_responsibilities_chunks(job_id)

    with open(f"./data/{job_id}/professional_exp_qwen.txt", "r") as fp:
        enhanced_cv_resp = fp.read()

    with open(f"./data/{job_id}/enhanced_skills_gpt.txt", "r") as fp:
        skills = json.load(fp)
        suggested_skills = [
            item["suggested_cv_skill"]
            for item in skills
            if item["category"] in ["Domain-Specific Skills", "Technical-Skills"]
        ]

    prompts = [
        {
            "role": "user",
            "content": f"""
                You are a helpful and expert assistant skilled in resume optimization and job matching. Your task is to revise a job applicant’s profile section based on a target job description. You must intelligently align the candidate’s past responsibilities with the responsibilities and required skills from the job description, while preserving the structure and truthfulness of the resume.

                When rewriting, follow these principles:

                Keep the structure of the CV intact.
                
                The profile section need to be around 4-5 lines of text, not more.

                Rewrite or reframe the responsibilities and achievements to better match the job description responsibilities.

                Use the most relevant technical and domain-specific skills from the job description where appropriate.

                You may omit less relevant responsibilities or reword them for alignment, but do not fabricate experience.

                Keep the tone professional and concise, using action verbs and impact-oriented phrasing.


                The output must only be the profile section without any explanation, intro, labeling.
                Here is the input data:

                1. Responsibilities from the job description:
                {jd_resp}
                \n
                2. Responsibilities from the applicant's resume aligned with responsibilities from the job description:
                {enhanced_cv_resp}
                \n
                3. Required technical and domain-specific skills:
                {suggested_skills}
                \n
                4. Profile section from the CV (to be rewritten):
                {cv_bio}
                \n
                Please rewrite the profile section to reflect the job description's responsibilities, Responsibilities from the applicant's resume, and preferred skills, maintaining factual accuracy and the original structure.
                """
        }
    ]

    response: ChatResponse = chat(model="qwen3:4b", messages=prompts)
    pattern = r'<think>.*?</think>'
    # Remove the think tags and their contents
    cleaned_text = re.sub(pattern, '', response["message"]["content"], flags=re.DOTALL)
    # Clean up any extra whitespace
    cleaned_text = cleaned_text.strip()
    with open(f"./data/{job_id}/profile_qwen.txt", "w") as fp:
        fp.write(cleaned_text)


def cv_bio_recommender_gpt(job_id):
    cv_bio = get_CV_bio(job_id)
    jd_resp = get_JD_responsibilities_chunks(job_id)

    with open(f"./data/{job_id}/professional_exp_gpt.txt", "r") as fp:
        enhanced_cv_resp = fp.read()

    with open(f"./data/{job_id}/enhanced_skills_gpt.txt", "r") as fp:
        skills = json.load(fp)
        suggested_skills = [
            item["suggested_cv_skill"]
            for item in skills
            if item["category"] in ["Domain-Specific Skills", "Technical-Skills"]
        ]

    prompts = [
        {
            "role": "system",
            "content": f"""
                You are a helpful and expert assistant skilled in resume optimization and job matching. Your task is to revise a job applicant’s profile section based on a target job description. You must intelligently align the candidate’s past responsibilities with the responsibilities and required skills from the job description, while preserving the structure and truthfulness of the resume.

                When rewriting, follow these principles:

                Keep the structure of the CV intact.

                The profile section need to be around 4-5 lines of text, not more.

                Rewrite or reframe the responsibilities and achievements to better match the job description responsibilities.

                Use the most relevant technical and domain-specific skills from the job description where appropriate.

                You may omit less relevant responsibilities or reword them for alignment, but do not fabricate experience.

                Keep the tone professional and concise, using action verbs and impact-oriented phrasing.


                The output must only be the profile section without any explanation, intro, labeling.
                """
        },
        {
            "role": "user",
            "content": f"""
            Here is the input data:

                1. Responsibilities from the job description:
                {jd_resp}
                \n
                2. Responsibilities from the applicant's resume aligned with responsibilities from the job description:
                {enhanced_cv_resp}
                \n
                3. Required technical and domain-specific skills:
                {suggested_skills}
                \n
                4. Profile section from the CV (to be rewritten):
                {cv_bio}
                \n
                Please rewrite the profile section to reflect the job description's responsibilities, Responsibilities from the applicant's resume, and preferred skills, maintaining factual accuracy and the original structure.
            """
        }
    ]

    openai = OpenAI(api_key=openai_api_key)
    response = openai.chat.completions.create(model="gpt-4o-mini", messages=prompts)
    with open(f"./data/{job_id}/profile_gpt.txt", "w") as fp:
        fp.write(response.choices[0].message.content)

    return response.choices[0].message.content


def cv_skills_format(job_id):
    with open(f"./data/{job_id}/enhanced_skills_gpt.txt", "r") as fp:
        skills = json.load(fp)

    prompts = [
        {
            "role": "system",
            "content": """
            
            """
        }
    ]


def format_skills_for_resume(job_id):
    with open(f"./data/{job_id}/enhanced_skills_gpt.txt", "r") as fp:
        skill_data = json.load(fp)
    # Initialize category buckets
    categorized_skills = defaultdict(list)

    # Collect skills per category
    for item in skill_data:
        skill = item["suggested_cv_skill"]
        category = item["category"]
        if skill and skill != "Not Found":
            categorized_skills[category].append(skill)

    # Define fixed order of categories
    category_order = ["Technical-Skills", "Domain-Specific Skills", "Soft-Skills"]

    # Build formatted output
    output_lines = []
    for category in category_order:
        skills = categorized_skills.get(category, [])
        if skills:
            # Remove duplicates and sort alphabetically
            unique_skills = set(skills)
            line = f"{' | '.join(unique_skills)}"
            output_lines.append(line)

    return output_lines[0], output_lines[1], output_lines[2]

def scoring_openai(job_id):
    print(f"Start parsing skills...")
    compute_skills_score_gpt(job_id)

    print(f"Start recommending skills...")
    skill_recommender_gpt(job_id)
    tech_skills, domain_skills,  soft_skills = format_skills_for_resume(job_id)

    print(f"Start recommending professional experiences...")
    responsibilities = responsibilities_recommender_gpt(job_id)

    print(f"Start recommending bio...")
    profile = cv_bio_recommender_gpt(job_id)

    return profile, responsibilities, tech_skills, domain_skills,  soft_skills


#compute_responsibilities_score()
#compute_skills_score()
#compute_skills_score_gpt(7375)
#compute_skills_score_llama()
#compute_skills_score_qwen()
#compute_skills_score()
#skill_recommender_gpt()
#skill_recommender_llama()
#skill_recommender_qwen()
#responsibilities_recommender_gpt()
#responsibilities_recommender_qwen()
#cv_bio_recommender_qwen()
#cv_bio_recommender_gpt()


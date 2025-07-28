from openai import OpenAI
from ollama import ChatResponse
from ollama import chat
import pickle
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "llama3.2:3b"

JOB_DESC = """
AI Engineer – GPT Search for Ecommerce


I'm working with one of Stockholm's most exciting and foreward thinking startups, a talented group of engineers & product-people who are looking for a cracked AI/ML engineer to join them as they transform the ecommerce space!


We're building a groundbreaking GPT-powered search experience that transforms traditional ecommerce search bars into intuitive, conversational interfaces. As a senior engineer, you’ll be at the core of this AI-native product, combining semantic search, RAG pipelines, and personalization to redefine how people discover products online.


🔧 What You’ll Do


Lead the development of a conversational search product powered by cutting-edge LLM technology.
Design, build, and optimize RAG pipelines, semantic retrieval systems, reranking models, and hyper-personalized search features.
Move fast: prototype rapidly, iterate based on real-world shopper and merchant feedback, and deploy to production.
Own the end-to-end ML lifecycle—from prompt engineering and model fine-tuning to evaluation and latency tuning.
Evaluate model architectures and APIs to balance performance, cost-efficiency, and user delight.
Collaborate cross-functionally with product, engineering, and leadership to craft a unified user experience.
Help scale a multi-tenant AI foundation across diverse brand needs with a mix of automation and customization.
Shape our engineering culture around experimentation, learning, and speed.


🙌 Who You Are
You think like a founder—scrappy, hands-on, and comfortable navigating ambiguity.
You’re energized by LLMs, vector databases, semantic search, and real-world AI applications.
You’ve built or contributed to production-grade ML systems—especially in ranking, retrieval, or inference-heavy domains.
You thrive in high-velocity environments where priorities evolve and builders lead the way.
You value impact over process and aren't afraid to hack something together to validate quickly.
Bonus if you’ve worked with ecommerce search, vector DBs (FAISS, Weaviate), LangChain, OpenAI APIs, or similar tools.


💼 What’s In It For You
Be a founding engineer of a high-impact product in a new AI category.
High ownership, full autonomy, and real influence on product direction.
Competitive compensation
Hybrid work - Stockholm HQ but with a very flexible model
30 days paid vacation + wellness and learning budgets..
Monthly co-working and quarterly off-site retreats.


This is a permanent role - Visa sponsorship is currently not supported.


Want to help reinvent how people shop online—powered by AI? This is your moment.
"""

JOB_DESC2 = """
Join our AI Technologies team at TRATON Group R&D

At TRATON Group, we believe that the whole can be greater than the sum of its parts. Together with our brands, we can make the future of transportation more sustainable - Let´s make a difference together. 

The TRATON AB office, located in Södertälje, consists of experienced colleagues with various backgrounds and nationalities from all TRATON Group brands. We enjoy solving strategic problems cross functional and cross brand in the TRATON Group. We strive for a climate where opinions and knowledge are openly shared within and between teams and we welcome new ideas in order to create dynamic synergies. 

With its brands Scania, MAN, Volkswagen Truck & Bus and International, TRATON Group is one of the world’s leading commercial vehicle manufacturers. Its offering comprises light-duty commercial vehicles, trucks, and buses. At TRATON, you are an important part of something bigger. Joining us means gaining access to the ins and outs of the entire transportation industry. As part of a global team of industry experts, you get to think bigger, experience more, and reach further. Being bigger also means being stronger.

Role summary

Are you passionate about creating and implementing deep learning data pipelines, CI/CD, cloud technology, and neural networks for autonomous driving? Are you eager to use academic knowledge and coding skills to solve real-world problems? Do you possess a strong technical background and thrive in a dynamic work environment? The Automated Driving department at TRATON Group R&D (Scania brand) is seeking a talented and motivated AI Engineer to join the AI Foundation team.

The team’s complex and fun mission is to seamlessly find a safe, comfortable, and eco-friendly trajectory for our autonomous vehicles. We consider various constraints, including the vehicle's physical limitations, traffic rules, and collision avoidance. Machine learning technologies can be of considerable assistance for this intricate challenge. Together with your team, you will research, develop, and create large-scale deep-learning data workflows and optimize models for performance and scalability. Join us and enjoy the ride together!

Your responsibilities

You will work with industrial research and pre-development in a collaborative and agile multi-cultural team within TRATON Group R&D, and your focus will be to develop, test, and run data pipelines in production at a large scale. The team’s responsibility spans from architectural designs, via software development to concept evaluation, as part of a more intelligent transportation system.

As part of this role, you will also:


Train, test, and deploy machine learning models on the cloud
Collaborate cross-functionally between teams to understand requirements and design optimal and robust data pipelines
Stay up to date with the very growing field of AI


Your profile

Our team is diverse with a mix of backgrounds and experiences. We are result-oriented, self-driven, and face new challenges with a positive, can-do attitude. To enjoy working with us we believe you are open and curious towards your teammates, communicative, can inspire others, supportive, and contribute to the development with your technical expertise.

Furthermore, we think you have the experience and knowledge in the following:


MSc/PhD in Computer Science, Machine Learning, Robotics, or related technical field of study
Implementing data pipelines to train deep learning models, integrating large-scale data workflows, and optimizing models for performance and scalability
Python and deep learning (using PyTorch or TensorFlow), as well as basic knowledge of SQL
Continuous Integration/Continuous Deployment (CI/CD) and unit/integration testing
Cloud technology such as EC2, IaM, infrastructure-as-code tools (IaC) such as Terraform
Distributed storage systems like AWS S3 and storage formats like Delta tables and Parquet.


Experience with the following is meritorious:


Distributed training of deep learning models
Foundation models, generative models, large language models (LLM), and visual language models (VLM).
Observability tools for logging and monitoring, such as Grafana and AWS CloudWatch.
Databricks and Apache Spark, including Spark SQL, DataFrame API, and Spark Streaming.
Data modeling concepts (Bronze, Silver, Gold layered architecture), ETL processes, performing data cleansing, and ensuring data quality throughout the pipeline.
Publications in top-tier conferences or journals related to machine learning


If you possess some (not necessarily all) of the requirements mentioned, we encourage you to apply with confidence. Every application is valued, and we welcome you to join us on this exciting journey of sustainability-driven innovation. Your unique skills and experience could be a perfect fit for our team!

We offer

At TRATON Group R&D Automated Driving, we want you to succeed and develop because together with your team you contribute to a sustainable future, securing a leading position for TRATON and the Scania brand in the industry.

Working at TRATON, you are offered benefits such as hybrid working, mutual performance bonus, flexible working hours, parental leave covered up to 90%, and much more.

For TRATON, diversity and inclusion are a strategic necessity. By having employees with the widest possible range of skills, knowledge, backgrounds, and experiences, we ensure we have the right people, and together with an inclusive corporate culture, this drives our business forward.

Contact information

For more information, you are welcome to contact Maria Linnarsson, Head of AI Technologies, at maria.linnarsson@scania.com. Please expect some delays in response during the summer vacation period.

Application

If you feel you have the skills and desire to take on this interesting and challenging role, please apply by answering the screening questions and submitting your CV, transcript of records, and relevant certificates. A background check might be conducted for this position. The last day to submit your application is 2025-08-12.

We are looking forward to reading your application!
"""


def extract_jd_components_llama(job_desc):
    response: ChatResponse = chat(model=MODEL_NAME, messages=[
        {
            "role": "user",
            "content": """
            You are a helpful assistant that extracts structured information from job descriptions. 

Your task is to read the following job description and extract the key details in JSON format.

### Instructions:
Carefully analyze the job description below and extract the following fields:

- job_title: (string) The full job title.
- seniority_level: (string) Junior, Mid, Senior, Lead, etc. If not mentioned, return "Not specified".
- required_skills: (list of strings) Hard/technical skills explicitly required.
- preferred_skills: (list of strings) Skills that are nice to have or preferred but not mandatory.
- soft_skills: (list of strings) Non-technical skills like communication, leadership, etc.
- responsibilities: (list of strings) Key responsibilities or duties for the role.
- qualifications: (list of strings) Required degrees, certifications, or education background.
- years_experience_required: (integer or string) Minimum years of relevant experience mentioned. If a range is given, use the minimum value. If not mentioned, return "Not specified".
- industry: (string) The relevant industry or domain (e.g., finance, healthcare, e-commerce). Return "Not specified" if unclear.
- employment_type: (string) Full-time, Part-time, Internship, Contract, etc. If not mentioned, return "Not specified".

### Output Format:
Return the results in a JSON format like this(no intro, no numbering, no bullets, just json):
{
  "job_title": "...",
  "seniority_level": "...",
  "required_skills": [...],
  "preferred_skills": [...],
  "soft_skills": [...],
  "responsibilities": [...],
  "qualifications": [...],
  "years_experience_required": ...,
  "industry": "...",
  "employment_type": "..."
}

### Job Description:
            \n
            """ + job_desc
        }
    ])

    print(response["message"]["content"])
    with open('./data/jd_llama.txt', 'w') as fp:
        fp.write(response["message"]["content"])



def extract_jd_components_openai(job_desc, job_id):
    print(f"Start parsing job ad...")
    openai = OpenAI(api_key=openai_api_key)
    system_prompt = """You are an expert at analyzing job descriptions and extracting key information. Your task is to carefully read job descriptions and extract structured information in JSON format.

    Your expertise includes:
    - Understanding various job description formats across different industries
    - Distinguishing between mandatory requirements and preferred qualifications
    - Identifying key responsibilities and role context
    - Extracting information even when it's scattered across different sections
    - Maintaining the original tone and specificity of the source material

    Always return valid JSON with no additional text or explanation."""


    user_prompt = """Extract the following information from the job description and return it as a valid JSON object:

                        ### Instructions:
Carefully analyze the job description below and extract the following fields:

- job_title: (string) The full job title.
- seniority_level: (string) Junior, Mid, Senior, Lead, etc. If not mentioned, return "Not specified".
- required_skills: (list of strings) Hard/technical skills explicitly required.
- preferred_skills: (list of strings) Skills that are nice to have or preferred but not mandatory.
- soft_skills: (list of strings) Non-technical skills like communication, leadership, etc.
- responsibilities: (list of strings) Key responsibilities or duties for the role.
- qualifications: (list of strings) Required degrees, certifications, or education background.
- years_experience_required: (integer or string) Minimum years of relevant experience mentioned. If a range is given, use the minimum value. If not mentioned, return "Not specified".
- industry: (string) The relevant industry or domain (e.g., finance, healthcare, e-commerce). Return "Not specified" if unclear.
- employment_type: (string) Full-time, Part-time, Internship, Contract, etc. If not mentioned, return "Not specified".

### Output Format:
Return the results in a JSON format like this:
{
  "job_title": "...",
  "seniority_level": "...",
  "required_skills": [...],
  "preferred_skills": [...],
  "soft_skills": [...],
  "responsibilities": [...],
  "qualifications": [...],
  "years_experience_required": ...,
  "industry": "...",
  "employment_type": "..."
}
                    
                        **Job Description to Analyze:**
                        """ + job_desc
    prompts = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompts,
        response_format={"type": "json_object"})

    print(response.choices[0].message.content)
    with open(f'./data/{job_id}/jd_gpt.txt', 'w') as fp:
        fp.write(response.choices[0].message.content)





#extract_jd_components_llama(JOB_DESC2)
#extract_jd_components_openai(JOB_DESC2)

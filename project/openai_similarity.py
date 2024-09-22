from openai import OpenAI
import os

# open_ai_access_key = os.environ.get("OPEN_AI_ACCESS_KEY")
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# OpenAI API key setup
load_dotenv()
client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

# OpenAI API key setup

def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

def calculate_similarity(candidate_text, job_text):
    candidate_embedding = get_embedding(candidate_text)
    job_embedding = get_embedding(job_text)

    candidate_embedding = np.array(candidate_embedding).reshape(1, -1)
    job_embedding = np.array(job_embedding).reshape(1, -1)

    similarity = cosine_similarity(candidate_embedding, job_embedding)[0][0]
    return similarity

def generate_summary(candidate, job):
    summary_text = (
        f"Candidate Name: {candidate['fullName']}\n"
        f"Skills: {', '.join(candidate['skills'])}\n"
        f"Experience: {', '.join([exp['jobTitle'] for exp in candidate['experience']])}\n"
        f"Why this candidate is a good fit for the job '{job['title']}': {job['description']}"
    )

    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": summary_text}
    ],
    max_tokens=100)
    return response.choices[0].message.content.strip()

def get_similarity_scores(candidates, job):
    job_text = f"Job Title: {job['title']}\nJob Description: {job['description']}\nRequired Skills: {job['requiredSkills']}"

    scores = []
    for candidate in candidates:
        candidate_text = f"Name: {candidate['fullName']}\nSkills: {', '.join(candidate['skills'])}\nExperience: {', '.join([exp['jobTitle'] for exp in candidate['experience']])}\nPreferred Location: {candidate['preferredLocation']}\nPreferred Job Type: {candidate['preferredJobType']}"
        similarity = calculate_similarity(candidate_text, job_text)

        # Match skills
        matched_skills = set(candidate['skills']) & set(job['requiredSkills'].split(", "))

        # Generate summary
        summary = generate_summary(candidate, job)

        scores.append({
            'candidateName': candidate['fullName'],
            'similarityScore': similarity,
            'matchedSkills': list(matched_skills),
            'summary': summary,
            'email': candidate['email'],
            'phone': candidate['phone']
        })

    scores = sorted(scores, key=lambda x: x['similarityScore'], reverse=True)

    return scores

# Sample input data
candidates = [
    {
        "_id": {"$oid": "66ef6890248ec1a4c28db987"},
        "fullName": "Michael Clark",
        "email": "michaelclark@example.com",
        "phone": "+1234567892",
        "skills": ["JavaScript", "Node.js", "React", "AWS"],
        "areaInterested": "Web Development",
        "experience": [
            {"jobTitle": "Full-Stack Developer", "company": "Web Solutions", "duration": "4 years", "description": "Developed and maintained web applications using React and Node.js."},
            {"jobTitle": "Frontend Developer", "company": "Creative Apps", "duration": "1.5 years", "description": "Created user-friendly UI/UX for web applications."}
        ],
        "preferredLocation": "Seattle, WA",
        "preferredJobType": "Full-time"
    },
    {
        "_id": {"$oid": "66ef6890248ec1a4c28db988"},
        "fullName": "Sara Thompson",
        "email": "sarathompson@example.com",
        "phone": "+1234567893",
        "skills": ["Python", "Data Analysis", "Machine Learning"],
        "areaInterested": "Data Science",
        "experience": [
            {"jobTitle": "Data Analyst", "company": "Data Insights", "duration": "3 years", "description": "Analyzed sales data to provide actionable insights."},
            {"jobTitle": "Junior Data Scientist", "company": "Tech Innovations", "duration": "2 years", "description": "Developed predictive models using Python."}
        ],
        "preferredLocation": "San Francisco, CA",
        "preferredJobType": "Remote"
    },
    {
        "_id": {"$oid": "66ef6890248ec1a4c28db989"},
        "fullName": "John Doe",
        "email": "johndoe@example.com",
        "phone": "+1234567894",
        "skills": ["Java", "Spring Boot", "Microservices"],
        "areaInterested": "Software Engineering",
        "experience": [
            {"jobTitle": "Backend Developer", "company": "Cloud Solutions", "duration": "5 years", "description": "Built and maintained microservices for cloud applications."},
            {"jobTitle": "Software Engineer", "company": "Web Tech", "duration": "2 years", "description": "Implemented RESTful APIs for web services."}
        ],
        "preferredLocation": "New York, NY",
        "preferredJobType": "Full-time"
    },
    {
        "_id": {"$oid": "66ef6890248ec1a4c28db990"},
        "fullName": "Emily Davis",
        "email": "emilydavis@example.com",
        "phone": "+1234567895",
        "skills": ["HTML", "CSS", "JavaScript", "React"],
        "areaInterested": "Web Development",
        "experience": [
            {"jobTitle": "Frontend Developer", "company": "Design Studio", "duration": "3 years", "description": "Created responsive web designs using React."},
            {"jobTitle": "Web Designer", "company": "Creative Solutions", "duration": "2 years", "description": "Designed user interfaces for various websites."}
        ],
        "preferredLocation": "Austin, TX",
        "preferredJobType": "Part-time"
    },
    {
        "_id": {"$oid": "66ef6890248ec1a4c28db991"},
        "fullName": "Robert Brown",
        "email": "robertbrown@example.com",
        "phone": "+1234567896",
        "skills": ["C++", "Embedded Systems", "IoT"],
        "areaInterested": "Embedded Systems",
        "experience": [
            {"jobTitle": "Embedded Systems Engineer", "company": "Tech Gadgets", "duration": "4 years", "description": "Designed embedded systems for smart devices."},
            {"jobTitle": "Junior Software Engineer", "company": "Innovation Labs", "duration": "1 year", "description": "Worked on firmware development for IoT devices."}
        ],
        "preferredLocation": "Chicago, IL",
        "preferredJobType": "Full-time"
    }
    # Additional candidates can be added here...
]

job = {
    "_id": {"$oid": "66ef6a5e248ec1a4c28db990"},
    "title": "Data Analyst",
    "description": "Analyze large datasets, create reports, and help the business make data-driven decisions.",
    "requiredSkills": "SQL, Python, PowerBI, Excel, Java, C, AWS, Node.js"
}

# Get similarity scores
similarity_scores = get_similarity_scores(candidates, job)

# Print the results
for score in similarity_scores:
    print(f"Candidate: {score['candidateName']}, Similarity Score: {score['similarityScore']:.4f}")
    print(f"Email: {score['email']}, Phone: {score['phone']}")
    print(f"Matched Skills: {', '.join(score['matchedSkills'])}")
    print(f"Summary: {score['summary']}\n")

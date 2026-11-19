"""Fresh jobs.csv generator. Run: python generate_jobs.py
Posting dates are generated relative to today, so re-running refreshes the dataset."""
import random
from datetime import date, timedelta
import pandas as pd

random.seed(2026)
TODAY = date.today()

ROLES = {
 "Python Developer": (["Python","Django","Flask","SQL","Git","REST API","Docker","PostgreSQL"], 0, 4, 5, 16),
 "Java Developer": (["Java","Spring Boot","SQL","Microservices","Git","Hibernate","Docker","Kafka"], 0, 6, 5, 20),
 "Frontend Developer": (["HTML","CSS","JavaScript","React","TypeScript","Redux","Tailwind","Git"], 0, 5, 4, 18),
 "Full Stack Developer": (["JavaScript","React","Node.js","MongoDB","SQL","TypeScript","Git","Docker"], 1, 6, 6, 24),
 "Backend Developer": (["Node.js","Express","MongoDB","SQL","REST API","Redis","Docker","AWS"], 0, 6, 5, 22),
 "Data Analyst": (["Python","SQL","Excel","Power BI","Tableau","Statistics","Pandas"], 0, 4, 4, 14),
 "Data Scientist": (["Python","Machine Learning","Pandas","Scikit-learn","SQL","Statistics","TensorFlow","NLP"], 1, 6, 9, 30),
 "Machine Learning Engineer": (["Python","Machine Learning","TensorFlow","PyTorch","Pandas","MLOps","Docker","AWS"], 1, 7, 10, 40),
 "AI Engineer": (["Python","LLM","PyTorch","NLP","LangChain","Docker","AWS","Deep Learning"], 1, 6, 12, 45),
 "Data Engineer": (["Python","SQL","Spark","Airflow","Kafka","AWS","ETL","Snowflake"], 1, 7, 8, 32),
 "DevOps Engineer": (["Docker","Kubernetes","AWS","Jenkins","Linux","Terraform","Git","CI/CD"], 1, 8, 7, 30),
 "Cloud Engineer": (["AWS","Azure","Terraform","Linux","Kubernetes","Python","Networking"], 1, 7, 8, 32),
 "Android Developer": (["Kotlin","Java","Android","Firebase","REST API","Git","Jetpack Compose"], 0, 6, 5, 22),
 "iOS Developer": (["Swift","SwiftUI","iOS","Xcode","REST API","Git","Firebase"], 1, 6, 7, 26),
 "Flutter Developer": (["Flutter","Dart","Firebase","REST API","Git","Android","iOS"], 0, 4, 5, 18),
 "C++ Developer": (["C++","Data Structures","Linux","Algorithms","Git","Multithreading"], 0, 6, 6, 26),
 "Golang Developer": (["Go","Microservices","Docker","Kubernetes","PostgreSQL","REST API","gRPC"], 1, 6, 9, 32),
 "Rust Developer": (["Rust","C++","Linux","Systems Programming","Git","WebAssembly"], 1, 6, 12, 38),
 ".NET Developer": (["C#",".NET","SQL Server","Azure","REST API","Git","Entity Framework"], 0, 7, 5, 22),
 "PHP Developer": (["PHP","Laravel","MySQL","JavaScript","HTML","CSS","Git"], 0, 4, 3, 12),
 "QA / Test Engineer": (["Selenium","Java","Python","Test Automation","Postman","SQL","Jira"], 0, 5, 4, 16),
 "Cyber Security Analyst": (["Network Security","SIEM","Python","Linux","Penetration Testing","Ethical Hacking"], 0, 6, 6, 28),
 "UI/UX Designer": (["Figma","UX Research","Wireframing","Prototyping","Adobe XD","HTML","CSS"], 0, 5, 4, 20),
 "Business Analyst": (["SQL","Excel","Power BI","Requirement Analysis","Jira","Communication","Tableau"], 0, 6, 5, 20),
 "Blockchain Developer": (["Solidity","Ethereum","JavaScript","Web3","Node.js","Smart Contracts"], 1, 5, 9, 34),
 "Game Developer": (["Unity","C#","C++","Unreal Engine","Game Design","Git"], 0, 5, 4, 20),
}
COMPANIES = ["TechNova","InsightIQ","NeuralWorks","PixelCraft","CloudNest","DataBridge","CodeCraft","InfoSys Labs",
 "Zenith Soft","QuantumLeap","BrightPath","AppForge","CyberShield","FinEdge","HealthPulse","EduSpark","ShopSphere",
 "LogiTrack","GreenGrid","MediaWave","Orbit Systems","Nimbus Tech","Pragati Digital","Swift Logic","Veda Analytics",
 "Kiran Labs","Aarav Software","Bharat Cloud","Indus AI","Vistara Tech"]
LOCS = ["Bangalore","Hyderabad","Pune","Noida","Gurgaon","Mumbai","Chennai","Delhi","Kolkata","Ahmedabad",
 "Jaipur","Chandigarh","Dehradun","Indore","Kochi","Remote"]
LOC_W = [20,12,11,10,8,9,8,6,3,3,2,2,2,2,2,14]
TYPES = ["Full-time","Part-time","Internship","Contract"]
DESC = ["Build and maintain scalable {r} solutions for production products",
 "Work with cross-functional teams as a {r} on high-impact projects",
 "Design, develop and ship features as a {r} in an agile team",
 "Own end-to-end delivery as a {r} and improve performance and quality",
 "Join our growing team as a {r} and work on modern tech stack"]

rows = []
for i in range(420):
    role = random.choice(list(ROLES))
    pool, emin, emax, smin, smax = ROLES[role]
    t = random.choices(TYPES, [70,8,12,10])[0]
    exp = 0 if t == "Internship" else random.randint(emin, emax)
    sal = round(random.uniform(smin, smax) * (0.25 if t == "Internship" else 1) * (0.6 if t == "Part-time" else 1) * (1 + exp * 0.08), 1)
    k = random.randint(4, min(6, len(pool)))
    skills = random.sample(pool, k)
    days = int(random.triangular(0, 45, 3))
    rows.append(dict(
        title=("Intern - " + role) if t == "Internship" else role,
        company=random.choice(COMPANIES), location=random.choices(LOCS, LOC_W)[0], type=t,
        experience=exp, salary_lpa=max(sal, 1.2), skills=";".join(skills),
        description=random.choice(DESC).format(r=role),
        posted_date=(TODAY - timedelta(days=days)).isoformat()))

df = pd.DataFrame(rows).drop_duplicates(subset=["title","company","location"]).reset_index(drop=True)
df.to_csv("jobs.csv", index=False)
print(len(df), "jobs saved; newest:", df.posted_date.max(), "| oldest:", df.posted_date.min())

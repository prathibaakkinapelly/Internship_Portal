from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import re

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Sample internship data matching your frontend structure
INTERNSHIP_DATA = [
    {
        "id": 1,
        "title": "Software Development Intern",
        "company": "TechCorp",
        "location": "Hyderabad",
        "stipend": 15000,
        "duration": "3 months",
        "required_skills": "Python, React, SQL, JavaScript",
        "description": "Work on cutting-edge web applications using modern technologies",
        "apply_link": "https://techcorp.com/careers",
        "company_logo_url": "https://via.placeholder.com/48x48?text=TC",
        "type": "Technology",
        "why": "Perfect match for your Python and React skills",
        "career_tip": "Focus on building full-stack projects to showcase your skills",
        "career_path": "Full Stack Developer → Senior Developer → Tech Lead"
    },
    {
        "id": 2,
        "title": "Data Science Intern",
        "company": "DataFlow Analytics",
        "location": "Bangalore",
        "stipend": 18000,
        "duration": "6 months",
        "required_skills": "Python, Machine Learning, Statistics, SQL",
        "description": "Analyze large datasets and build predictive models",
        "apply_link": "https://dataflow.com/internships",
        "company_logo_url": "https://via.placeholder.com/48x48?text=DF",
        "type": "Data Science",
        "why": "Your Python skills align perfectly with our data pipeline requirements",
        "career_tip": "Master pandas, numpy, and scikit-learn for data manipulation",
        "career_path": "Data Analyst → Data Scientist → Senior Data Scientist"
    },
    {
        "id": 3,
        "title": "Digital Marketing Intern",
        "company": "MarketPro Solutions",
        "location": "Mumbai",
        "stipend": 12000,
        "duration": "4 months",
        "required_skills": "Social Media Marketing, Content Writing, Google Analytics, SEO",
        "description": "Create engaging marketing campaigns and analyze performance",
        "apply_link": "https://marketpro.com/jobs",
        "company_logo_url": "https://via.placeholder.com/48x48?text=MP",
        "type": "Marketing",
        "why": "Great opportunity to combine creativity with data-driven marketing",
        "career_tip": "Learn Google Ads and Facebook Ads for better job prospects",
        "career_path": "Marketing Associate → Marketing Manager → CMO"
    },
    {
        "id": 4,
        "title": "Frontend Developer Intern",
        "company": "WebSolutions Inc",
        "location": "Remote",
        "stipend": 20000,
        "duration": "3 months",
        "required_skills": "JavaScript, React, HTML, CSS, Git",
        "description": "Build responsive and interactive user interfaces",
        "apply_link": "https://websolutions.com/careers",
        "company_logo_url": "https://via.placeholder.com/48x48?text=WS",
        "type": "Technology",
        "why": "Your React experience makes you an ideal candidate",
        "career_tip": "Master modern CSS frameworks like Tailwind CSS",
        "career_path": "Frontend Developer → Senior Frontend → Full Stack Developer"
    },
    {
        "id": 5,
        "title": "Backend Developer Intern",
        "company": "ServerTech",
        "location": "Pune",
        "stipend": 16000,
        "duration": "4 months",
        "required_skills": "Python, Django, PostgreSQL, REST APIs",
        "description": "Develop scalable backend systems and APIs",
        "apply_link": "https://servertech.com/internships",
        "company_logo_url": "https://via.placeholder.com/48x48?text=ST",
        "type": "Technology",
        "why": "Strong Python background fits our backend requirements perfectly",
        "career_tip": "Learn Docker and cloud services like AWS",
        "career_path": "Backend Developer → Senior Backend → System Architect"
    },
    {
        "id": 6,
        "title": "Mobile App Development Intern",
        "company": "AppCrafters",
        "location": "Chennai",
        "stipend": 14000,
        "duration": "5 months",
        "required_skills": "React Native, JavaScript, Mobile UI/UX",
        "description": "Build cross-platform mobile applications",
        "apply_link": "https://appscrafters.com/jobs",
        "company_logo_url": "https://via.placeholder.com/48x48?text=AC",
        "type": "Technology",
        "why": "Your React skills translate perfectly to React Native development",
        "career_tip": "Learn native iOS/Android development for better opportunities",
        "career_path": "Mobile Developer → Senior Mobile Dev → Mobile Architect"
    },
    {
        "id": 7,
        "title": "DevOps Intern",
        "company": "CloudOps",
        "location": "Hyderabad",
        "stipend": 17000,
        "duration": "6 months",
        "required_skills": "Linux, Docker, AWS, Python, Git",
        "description": "Automate deployment pipelines and manage cloud infrastructure",
        "apply_link": "https://cloudops.com/careers",
        "company_logo_url": "https://via.placeholder.com/48x48?text=CO",
        "type": "Technology",
        "why": "Python scripting skills are valuable for automation tasks",
        "career_tip": "Get AWS certifications to boost your DevOps career",
        "career_path": "DevOps Engineer → Senior DevOps → Cloud Architect"
    },
    {
        "id": 8,
        "title": "UI/UX Design Intern",
        "company": "DesignStudio",
        "location": "Bangalore",
        "stipend": 13000,
        "duration": "4 months",
        "required_skills": "Figma, Adobe XD, User Research, Prototyping",
        "description": "Design intuitive user interfaces and experiences",
        "apply_link": "https://designstudio.com/internships",
        "company_logo_url": "https://via.placeholder.com/48x48?text=DS",
        "type": "Design",
        "why": "Great opportunity to combine technical knowledge with design thinking",
        "career_tip": "Build a strong portfolio showcasing your design process",
        "career_path": "UI/UX Designer → Senior Designer → Design Lead"
    },
    {
        "id": 9,
        "title": "Cybersecurity Intern",
        "company": "SecureNet",
        "location": "Delhi",
        "stipend": 19000,
        "duration": "6 months",
        "required_skills": "Network Security, Python, Ethical Hacking, Linux",
        "description": "Learn and implement cybersecurity best practices",
        "apply_link": "https://securenet.com/careers",
        "company_logo_url": "https://via.placeholder.com/48x48?text=SN",
        "type": "Security",
        "why": "Python scripting skills are essential for security automation",
        "career_tip": "Get certified in ethical hacking (CEH) and security frameworks",
        "career_path": "Security Analyst → Security Engineer → CISO"
    },
    {
        "id": 10,
        "title": "Content Writing Intern",
        "company": "ContentHub",
        "location": "Remote",
        "stipend": 10000,
        "duration": "3 months",
        "required_skills": "Content Writing, SEO, WordPress, Research",
        "description": "Create engaging content for blogs and social media",
        "apply_link": "https://contenthub.com/jobs",
        "company_logo_url": "https://via.placeholder.com/48x48?text=CH",
        "type": "Content",
        "why": "Good opportunity to develop communication skills alongside technical expertise",
        "career_tip": "Learn technical writing to combine with your technical background",
        "career_path": "Content Writer → Content Manager → Content Strategy Lead"
    }
]

class InternshipRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.scaler = StandardScaler()
        self.knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.feature_matrix = None
        self.is_fitted = False
        
    def preprocess_skills(self, skills_text):
        """Clean and preprocess skills text"""
        if not skills_text:
            return ""
        return re.sub(r'[^a-zA-Z\s+#]', ' ', skills_text.lower())
    
    def calculate_skill_similarity(self, user_skills, job_skills):
        """Calculate skill similarity using TF-IDF"""
        if not user_skills or not job_skills:
            return 0.0
        
        user_skills_clean = self.preprocess_skills(user_skills)
        job_skills_clean = self.preprocess_skills(job_skills)
        
        user_skill_set = set(user_skills_clean.split())
        job_skill_set = set(job_skills_clean.split())
        
        if not user_skill_set or not job_skill_set:
            return 0.0
            
        intersection = len(user_skill_set.intersection(job_skill_set))
        union = len(user_skill_set.union(job_skill_set))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_location_score(self, user_location, job_location):
        """Calculate location match score"""
        if not user_location or not job_location:
            return 0.5
        
        user_location = user_location.lower().strip()
        job_location = job_location.lower().strip()
        
        if 'remote' in job_location:
            return 1.0
        if user_location in job_location or job_location in user_location:
            return 1.0
        
        # Check for same state/city matches
        user_parts = user_location.replace(',', ' ').split()
        job_parts = job_location.replace(',', ' ').split()
        
        common_parts = set(user_parts).intersection(set(job_parts))
        if common_parts:
            return 0.8
            
        return 0.3
    
    def calculate_stipend_score(self, user_stipend, job_stipend):
        """Calculate stipend compatibility score"""
        if not user_stipend:
            return 0.5
        
        try:
            user_stipend = float(user_stipend)
            job_stipend = float(job_stipend)
            
            if job_stipend >= user_stipend:
                return 1.0
            else:
                ratio = job_stipend / user_stipend
                return max(0.2, ratio)
        except:
            return 0.5
    
    def create_feature_vector(self, user_data):
        """Create feature vector for user"""
        features = []
        
        # Skill similarity scores
        for internship in INTERNSHIP_DATA:
            skill_score = self.calculate_skill_similarity(
                user_data.get('skills', ''), 
                internship['required_skills']
            )
            features.append(skill_score)
        
        # Location scores
        for internship in INTERNSHIP_DATA:
            location_score = self.calculate_location_score(
                user_data.get('preferred_location', ''),
                internship['location']
            )
            features.append(location_score)
        
        # Stipend scores
        for internship in INTERNSHIP_DATA:
            stipend_score = self.calculate_stipend_score(
                user_data.get('stipend_expected'),
                internship['stipend']
            )
            features.append(stipend_score)
        
        return np.array(features).reshape(1, -1)
    
    def fit(self):
        """Prepare the model with internship data"""
        # Create feature matrix for all internships
        features = []
        
        for i, internship in enumerate(INTERNSHIP_DATA):
            feature_vector = []
            
            # Self-similarity for skills (always 1.0)
            for j, other_internship in enumerate(INTERNSHIP_DATA):
                if i == j:
                    feature_vector.append(1.0)
                else:
                    skill_sim = self.calculate_skill_similarity(
                        internship['required_skills'],
                        other_internship['required_skills']
                    )
                    feature_vector.append(skill_sim)
            
            # Location self-matches
            for j, other_internship in enumerate(INTERNSHIP_DATA):
                if i == j:
                    feature_vector.append(1.0)
                else:
                    loc_sim = 1.0 if internship['location'].lower() == other_internship['location'].lower() else 0.5
                    feature_vector.append(loc_sim)
            
            # Stipend similarities
            for j, other_internship in enumerate(INTERNSHIP_DATA):
                if i == j:
                    feature_vector.append(1.0)
                else:
                    stipend_sim = min(internship['stipend'], other_internship['stipend']) / max(internship['stipend'], other_internship['stipend'])
                    feature_vector.append(stipend_sim)
            
            features.append(feature_vector)
        
        self.feature_matrix = np.array(features)
        self.knn.fit(self.feature_matrix)
        self.is_fitted = True
    
    def get_recommendations(self, user_data):
        """Get KNN-based recommendations with scoring"""
        if not self.is_fitted:
            self.fit()
        
        # Create user feature vector
        user_vector = self.create_feature_vector(user_data)
        
        # Get KNN recommendations
        distances, indices = self.knn.kneighbors(user_vector)
        
        recommendations = []
        
        for i, idx in enumerate(indices[0]):
            internship = INTERNSHIP_DATA[idx].copy()
            
            # Calculate detailed scores
            skill_score = self.calculate_skill_similarity(
                user_data.get('skills', ''), 
                internship['required_skills']
            )
            location_score = self.calculate_location_score(
                user_data.get('preferred_location', ''),
                internship['location']
            )
            stipend_score = self.calculate_stipend_score(
                user_data.get('stipend_expected'),
                internship['stipend']
            )
            
            # Overall score with weights
            overall_score = (
                skill_score * 0.5 +
                location_score * 0.3 +
                stipend_score * 0.2
            )
            
            # KNN distance score (lower distance = higher similarity)
            knn_score = max(0, 1 - distances[0][i])
            
            # Combined score
            final_score = (overall_score * 0.7) + (knn_score * 0.3)
            
            internship.update({
                'score': final_score,
                'match_percentage': round(final_score * 100, 1),
                'skill_match': round(skill_score, 2),
                'location_match': round(location_score, 2),
                'stipend_match': round(stipend_score, 2),
                'knn_distance': round(distances[0][i], 3)
            })
            
            recommendations.append(internship)
        
        # Sort by final score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:5]

# Initialize the recommender
recommender = InternshipRecommender()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """API endpoint for getting internship recommendations"""
    try:
        # Get form data
        user_data = {
            'name': request.form.get('name', ''),
            'college_name': request.form.get('college_name', ''),
            'college_area': request.form.get('college_area', ''),
            'graduation_year': request.form.get('graduation_year', ''),
            'skills': request.form.get('skills', ''),
            'interests': request.form.get('interests', ''),
            'preferred_location': request.form.get('preferred_location', ''),
            'stipend_expected': request.form.get('stipend_expected'),
            'language': request.form.get('language', 'en')
        }
        
        # Handle file upload
        resume_filename = None
        if 'resume' in request.files:
            file = request.files['resume']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                resume_filename = f"{timestamp}_{filename}"
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], resume_filename))
                user_data['resume_filename'] = resume_filename
        
        # Validate required fields
        if not user_data['name'] or not user_data['skills']:
            return jsonify({'error': 'Name and skills are required fields'}), 400
        
        # Get KNN-based recommendations
        recommendations = recommender.get_recommendations(user_data)
        
        # Prepare response
        response_data = {
            'user': user_data,
            'results': recommendations,
            'total_matches': len(INTERNSHIP_DATA),
            'timestamp': datetime.now().isoformat(),
            'algorithm_used': 'KNN + Scoring'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in recommend endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error. Please try again.'}), 500

@app.route('/api/internships')
def get_all_internships():
    """Get all available internships"""
    return jsonify({
        'internships': INTERNSHIP_DATA,
        'total': len(INTERNSHIP_DATA)
    })

@app.route('/api/internship/<int:internship_id>')
def get_internship_details(internship_id):
    """Get details of a specific internship"""
    internship = next((item for item in INTERNSHIP_DATA if item["id"] == internship_id), None)
    if internship:
        return jsonify(internship)
    else:
        return jsonify({'error': 'Internship not found'}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request. Please check your input.'}), 400

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error. Please try again later.'}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("Starting Internship Recommendation Portal with KNN Algorithm...")
    print("Available endpoints:")
    print("  GET  /                    - Main application")
    print("  POST /api/recommend       - Get KNN recommendations")
    print("  GET  /api/internships     - Get all internships")
    print("  GET  /api/internship/<id> - Get specific internship")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
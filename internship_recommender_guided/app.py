
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import os
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import re

app = Flask(__name__)
CORS(app)

# Test endpoint to check database connection
@app.route('/api/db_test')
def db_test():
    try:
        conn = sqlite3.connect('Database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT name FROM sqlite_master WHERE type="table";')
        tables = cursor.fetchall()
        conn.close()
        return jsonify({'status': 'success', 'tables': tables})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

app = Flask(__name__)
CORS(app)


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
        except Exception:
            return 0.5
    
    def fetch_internships(self):
            try:
                conn = sqlite3.connect('Database.db')
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM internships1')
                columns = [desc[0] for desc in cursor.description]
                internships = [dict(zip(columns, row)) for row in cursor.fetchall()]
                conn.close()
                print(f"DEBUG: Fetched {len(internships)} internships from database.")
                return internships
            except Exception as e:
                print(f"ERROR: Database connection or query failed: {e}")
                return []

    def create_feature_vector(self, user_data, internships):
        """Create feature vector for user"""
        features = []
        
        for internship in internships:
            skill_score = self.calculate_skill_similarity(
                user_data.get('skills', ''), 
                internship['required_skills']
            )
            features.append(skill_score)
        
        for internship in internships:
            location_score = self.calculate_location_score(
                user_data.get('preferred_location', ''),
                internship['location']
            )
            features.append(location_score)
        
        for internship in internships:
            stipend_score = self.calculate_stipend_score(
                user_data.get('stipend_expected'),
                internship['stipend']
            )
            features.append(stipend_score)
        return np.array(features).reshape(1, -1)
    
    def fit(self, internships):
        """Prepare the model with internship data"""
        features = []
        for i, internship in enumerate(internships):
            feature_vector = []
            
            for j, other_internship in enumerate(internships):
                if i == j:
                    feature_vector.append(1.0)
                else:
                    skill_sim = self.calculate_skill_similarity(
                        internship['required_skills'],
                        other_internship['required_skills']
                    )
                    feature_vector.append(skill_sim)
            
            for j, other_internship in enumerate(internships):
                if i == j:
                    feature_vector.append(1.0)
                else:
                    loc_sim = 1.0 if internship['location'].lower() == other_internship['location'].lower() else 0.5
                    feature_vector.append(loc_sim)
            
            for j, other_internship in enumerate(internships):
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
        internships = self.fetch_internships()
        if not self.is_fitted:
            self.fit(internships)
        
        user_vector = self.create_feature_vector(user_data, internships)
        
        distances, indices = self.knn.kneighbors(user_vector)
        recommendations = []
        for i, idx in enumerate(indices[0]):
            internship = internships[idx].copy()
            
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
            
            overall_score = (
                skill_score * 0.5 +
                location_score * 0.3 +
                stipend_score * 0.2
            )
            
            knn_score = max(0, 1 - distances[0][i])
            
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
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:5]


recommender = InternshipRecommender()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """API endpoint for getting internship recommendations"""
    try:
        
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
        
        resume_filename = None
        if 'resume' in request.files:
            file = request.files['resume']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                resume_filename = f"{timestamp}_{filename}"
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], resume_filename))
                user_data['resume_filename'] = resume_filename
    
        if not user_data['name'] or not user_data['skills']:
            return jsonify({'error': 'Name and skills are required fields'}), 400
        
        
        recommendations = recommender.get_recommendations(user_data)
        print('DEBUG: Recommendations:', recommendations)
        
        response_data = {
            'user': user_data,
            'results': recommendations,
            'total_matches': len(recommender.fetch_internships()),
            'timestamp': datetime.now().isoformat(),
            'algorithm_used': 'KNN + Scoring'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in recommend endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error. Please try again.'}), 500

@app.route('/api/internships')
def get_all_internships():
    """Get all available internships from Database.db"""
    conn = sqlite3.connect('Database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM internships1')
    columns = [desc[0] for desc in cursor.description]
    internships = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return jsonify({
        'internships': internships,
        'total': len(internships)
    })

@app.route('/api/internship/<int:internship_id>')
def get_internship_details(internship_id):
    """Get details of a specific internship from data.db"""
    conn = sqlite3.connect('Database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM internships1 WHERE id = ?', (internship_id,))
    row = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]
    conn.close()
    if row:
        internship = dict(zip(columns, row))
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
    
    os.makedirs('templates', exist_ok=True)
    
    print("Starting Internship Recommendation Portal with KNN Algorithm...")
    print("Available endpoints:")
    print("  GET  /                    - Main application")
    print("  POST /api/recommend       - Get KNN recommendations")
    print("  GET  /api/internships     - Get all internships")
    print("  GET  /api/internship/<id> - Get specific internship")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
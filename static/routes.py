from flask import render_template, request,redirect,url_for,flash,jsonify
from flask_login import login_user, logout_user, current_user, login_required
import google.generativeai as genai
from models import User



genai.configure(api_key="AIzaSyDn4MtF99w6vIP2lafquqUGSET63pHg6B0")

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)




def register_routes(app, db, bcrypt):

    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/contact')
    def contact():
        return render_template('contact.html')
    
    @app.route('/contribution')
    def contribution():
        return render_template('contribution.html')
    
    @app.route('/notes')
    def notes():
        return render_template('create_new.html')
    

    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        if request.method == 'GET':
            return render_template('signup.html')
        elif request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            role = request.form['role']
            gender = request.form.get('gender')  # This can be optional
            mobile = request.form.get('mobile')  # This can be optional
            email = request.form['email']
            if not username or not password or not role or not email:
                flash('All fields except gender and mobile are required.', 'danger')
                return redirect(url_for('signup'))

            hashed_password = bcrypt.generate_password_hash(password.encode('utf-8'))
            user = User(
                username=username, 
                password=hashed_password, 
                role=role,
                gender=gender, 
                mobile=mobile, 
                email=email
            )
            # db.session.add(user)
            # db.session.commit()
            try:
            # Add and commit the user to the database
                db.session.add(user)
                db.session.commit()
                flash('Account created successfully! You can now log in.', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                db.session.rollback()  # Rollback in case of an error
                flash('An error occurred while creating your account. Please try again.', 'danger')
                return redirect(url_for('signup'))

    
        
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'GET':
            return render_template('login.html')
        elif request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            user = User.query.filter(User.username == username).first()
            if (user and bcrypt.check_password_hash(user.password, password.encode('utf-8'))):
                login_user(user)
                return redirect(url_for('index'))
            else:
                return render_template('404.html')
    

    
    @app.route('/logout')
    def logout():
        logout_user()
        return redirect(url_for('index'))
    
    @app.route('/summarize', methods=['POST'])
    def summarize():
        text = request.json['text']
        try:
            chat_session = model.start_chat(history=[])
            response = chat_session.send_message(f"Summarize the following text: {text}")
            summary = response.text
            return jsonify({'summary': summary})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    @app.route('/summary')
    def summary_page():
        summary = request.args.get('summary', '')
        return render_template('summary.html', summary=summary)
    






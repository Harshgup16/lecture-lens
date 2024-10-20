from flask import render_template, request,redirect,url_for,flash,jsonify
from flask_login import login_user, logout_user, current_user, login_required
import google.generativeai as genai
from models import User



genai.configure(api_key="Your_API_Key")

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





from flask import render_template, request, redirect, url_for, flash, jsonify
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio
import numpy as np
import tempfile
import os

# Initialize Whisper model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
whisper_model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)
def generate_notes(transcription):
    prompt = """
    You are an AI assistant specialized in organizing information into a book-like structure. Your task is to take the provided text input and restructure it into a format resembling a book chapter or section. Follow these guidelines:

    1. Identify the main topic from the input text.
    2. Create a title for this main topic.
    3. Break down the main topic into relevant subtopics.
    4. For each subtopic:
        a. Provide a clear, concise heading.
        b. Include a brief explanation or key points related to the subtopic.
        c. If necessary, add more detailed information or examples.
    5. Ensure a logical flow between subtopics.
    6. Use appropriate formatting to distinguish between titles, subtopics, and explanations.

    Your output should follow this general structure:

    # Main Topic Title

    ## Subtopic 1
    Brief explanation of Subtopic 1
    - Key point 1
    - Key point 2

    Additional details or examples if needed.

    ## Subtopic 2
    Brief explanation of Subtopic 2
    - Key point 1
    - Key point 2

    Additional details or examples if needed.

    [Continue with more subtopics as necessary]

    Input: {transcription}

    Please process the input text and provide a well-structured, book-like output following the guidelines above.
    """

    response = model.generate_content(prompt.format(transcription=transcription))
    return response.text


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
            response = chat_session.send_message(f'''From now - generate detailed and organized notes from transcribe provided, following a structured format: start with a concise title summarizing the lecture topic, followed by a subtitle providing context; include a list of important keywords and an index of main topics for easy navigation; define critical terminologies covered; develop the content by explaining each main topic with relevant subpoints, examples, and any formulas mentioned; use bullet points for key lists or steps, and reference links if applicable; conclude with a summary of key takeaways, ensuring the final output is precise, logically structured, and easy to review.''')
            response = chat_session.send_message(f'''if the teacher uses vague language, apply context-based analysis to infer the most likely meaning. If the meaning cannot be determined with sufficient confidence, omit the unclear segment to maintain clarity.Include a disclaimer where sections were excluded or inferred due to vague language, ensuring that the final output remains coherent, accurate, and satisfactory for study purposes.''')
            response = chat_session.send_message(f'''Analyze the transcribed lecture and provide response in html format for transcrpt as following - {text}''')
            summary = response.text
            return jsonify({'summary': summary})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    @app.route('/summary')
    def summary_page():
        summary = request.args.get('summary', '')
        return render_template('summary.html', summary=summary)
    

    @app.route('/process-audio', methods=['POST'])
    def process_audio():
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Use a context manager for the temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_file_path = tmp_file.name
        
        try:
            # Load and process audio
            waveform, sample_rate = torchaudio.load(tmp_file_path)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0).unsqueeze(0)
            
            waveform = waveform.numpy()
            waveform = np.squeeze(waveform)
            
            # Get transcription
            result = pipe(waveform, return_timestamps=True, generate_kwargs={"task": "transcribe"})
            
            # Generate notes using Gemini API
            notes = generate_notes(result["text"])
            
            return jsonify({
                'transcription': result["text"],
                'timestamps': result.get("timestamps", []),
                'notes': notes
            })
            
        except Exception as e:
            app.logger.error(f"Error processing audio: {str(e)}")
            return jsonify({'error': 'An error occurred while processing the audio'}), 500
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except PermissionError:
                app.logger.warning(f"Could not delete temporary file: {tmp_file_path}")
            except Exception as e:
                app.logger.error(f"Error deleting temporary file: {str(e)}")

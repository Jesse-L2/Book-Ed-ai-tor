"""
BookAIditor Web Server - Flask web interface for analyzing and improving writing quality
This script provides a web interface to the BookAIditor functionality, allowing users to
upload Word documents, process them through Ollama with Gemma 3, and receive
streamed writing improvement suggestions in the browser.
"""

import os
import json
import time
import docx
import threading
from flask import Flask, request, render_template, jsonify, Response, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests
from tqdm import tqdm

# Import from the original BookAIditor script
from contexts.prompt_contexts import (
    literary_fiction_context,
    thriller_context,
    romance_context,
    sci_fi_context,
    young_adult_context,
    fantasy_context,
    horror_context,
    mystery_context
)

app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'docx'}
DEFAULT_MODEL = 'gemma3:1b'
DEFAULT_CHUNK_SIZE = 3
DEFAULT_API_URL = 'http://localhost:11434'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Create necessary folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

# Create a templates folder and add the HTML templates there
os.makedirs('templates', exist_ok=True)

# Track active analysis jobs
active_jobs = {}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_word_document(file_path):
    """Read content from a Word document file."""
    try:
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return doc, paragraphs
    except Exception as e:
        print(f"Error reading document: {e}")
        return None, []

def chunk_paragraphs(paragraphs, chunk_size):
    """Split paragraphs into chunks for processing."""
    for i in range(0, len(paragraphs), chunk_size):
        yield paragraphs[i:i + chunk_size]

def get_additional_context(genre):
    """Determine additional injected prompt context based on genre."""
    context_map = {
        'literary_fiction': literary_fiction_context,
        'thriller': thriller_context,
        'romance': romance_context,
        'sci_fi': sci_fi_context,
        'young_adult': young_adult_context,
        'fantasy': fantasy_context,
        'horror': horror_context,
        'mystery': mystery_context
    }
    
    return context_map.get(genre, "")

def analyze_text_with_ollama(text_chunk, model_name, api_url, additional_context=""):
    """
    Send text to Ollama API for analysis and improvement suggestions.
    """
    prompt = f"""
You are a professional book editor with years of experience. You have a deep knowledge of audience expectations, understand tropes, and can identify writing quality issues. You are tasked with analyzing and improving the writing quality and marketability of a passage. Your feedback should be constructive and actionable. Your feedback should tighten language, improve clarity, enhance rhythm, all while preserving the author's voice and intent. You will enhance the narrative logic and flow, ensuring the text is engaging and easy to read. You will suggest improvements to character development, pacing, and plot structure where necessary.

Additional context may or may not be provided. Additional context: {additional_context}

Analyze the following text passage and provide detailed feedback:

{text_chunk}

For each paragraph, provide:
1. Writing quality issues (wordiness, passive voice, unclear sentences, etc.)
2. Suggested improvements (clearer wording, stronger verbs, better sentence structure)
3. An improved version of the paragraph

Format your response as a JSON object with these keys:
- "overall_assessment": General thoughts on the writing quality
- "paragraphs": Array of objects, each containing:
- "original": The original paragraph
- "issues": Array of identified issues
- "suggestions": Array of improvement suggestions
- "improved_version": Edited paragraph with improvements applied

Only respond with valid JSON - nothing else before or after.
"""

    try:
        response = requests.post(
            f"{api_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                }
            },
            timeout=120
        )
        
        if response.status_code == 200:
            response_text = response.json().get('response', '')
            
            # Extract JSON from response (in case there's any text before/after)
            try:
                # Find JSON content - look for first { and last }
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_content = response_text[json_start:json_end]
                    return json.loads(json_content)
                else:
                    print("Could not find JSON content in response")
                    return None
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON response: {e}")
                print(f"Raw response: {response_text[:200]}...")
                return None
        else:
            print(f"API request failed with status {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Error during API request: {e}")
        return None

def create_improved_document(original_doc, paragraphs, analysis_results):
    """Create a new document with original text, issues, and suggestions."""
    new_doc = docx.Document()
    
    # Add title
    new_doc.add_heading('Writing Improvement Suggestions', 0)
    
    # Add overall assessment section
    new_doc.add_heading('Overall Assessment', 1)
    overall_assessment = "No overall assessment available."
    
    if analysis_results and len(analysis_results) > 0:
        # Get the overall assessment from the first chunk that has one
        for result in analysis_results:
            if result and 'overall_assessment' in result:
                overall_assessment = result['overall_assessment']
                break
    
    new_doc.add_paragraph(overall_assessment)
    
    # Add detailed analysis for each paragraph
    new_doc.add_heading('Detailed Analysis', 1)
    
    paragraph_index = 0  # Reset paragraph index to track all paragraphs globally
    for chunk_result in analysis_results:
        if not chunk_result or 'paragraphs' not in chunk_result:
            # Skip failed analyses
            continue
            
        for para_analysis in chunk_result.get('paragraphs', []):
            if paragraph_index >= len(paragraphs):
                break
                
            # Section for this paragraph
            new_doc.add_heading(f'Paragraph {paragraph_index + 1}', 2)
            
            # Original text
            original_heading = new_doc.add_paragraph('Original Text: ')
            original_heading.runs[0].bold = True
            new_doc.add_paragraph(paragraphs[paragraph_index])
            
            # Issues
            if 'issues' in para_analysis and para_analysis['issues']:
                issues_heading = new_doc.add_paragraph('Issues: ')
                issues_heading.runs[0].bold = True
                for issue in para_analysis['issues']:
                    new_doc.add_paragraph(f"• {issue}")
            
            # Suggestions
            if 'suggestions' in para_analysis and para_analysis['suggestions']:
                suggestions_heading = new_doc.add_paragraph('Suggestions: ')
                suggestions_heading.runs[0].bold = True
                for suggestion in para_analysis['suggestions']:
                    new_doc.add_paragraph(f"• {suggestion}")
            
            # Improved version
            if 'improved_version' in para_analysis:
                improved_heading = new_doc.add_paragraph('Improved Version: ')
                improved_heading.runs[0].bold = True
                improved_para = new_doc.add_paragraph()
                improved_run = improved_para.add_run(para_analysis['improved_version'])
                improved_run.font.color.theme_color = 1  # Use theme color (usually blue)
            
            new_doc.add_paragraph('---')  # Separator
            paragraph_index += 1  # Increment global paragraph index
    
    return new_doc

def process_document(job_id, file_path, model_name, chunk_size, api_url, genre, sample_only=False):
    """Process a document and stream results"""
    # Read the document
    doc, paragraphs = read_word_document(file_path)
    if not doc:
        yield json.dumps({"error": "Failed to read document"})
        return

    # Get additional context based on genre
    additional_context = get_additional_context(genre)
    
    # Create chunks of paragraphs
    chunks = list(chunk_paragraphs(paragraphs, chunk_size))
    if sample_only:
        chunks = chunks[:1]
    
    # Store total chunks for progress calculations
    active_jobs[job_id]["total_chunks"] = len(chunks)
    active_jobs[job_id]["processed_chunks"] = 0
    active_jobs[job_id]["analysis_results"] = []
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        chunk_text = "\n\n".join(chunk)
        
        # Update status for frontend
        progress_info = {
            "status": "processing",
            "message": f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} paragraphs)",
            "progress": (i / len(chunks)) * 100
        }
        yield json.dumps(progress_info)
        
        # Analyze chunk
        result = analyze_text_with_ollama(chunk_text, model_name, api_url, additional_context)
        
        # Save the result
        active_jobs[job_id]["analysis_results"].append(result)
        active_jobs[job_id]["processed_chunks"] += 1
        
        # Send chunk result to frontend
        if result:
            chunk_result = {
                "status": "chunk_complete",
                "chunk_index": i,
                "chunk_result": result,
                "progress": ((i + 1) / len(chunks)) * 100
            }
            yield json.dumps(chunk_result)
        else:
            error_info = {
                "status": "chunk_error",
                "message": f"Failed to analyze chunk {i+1}",
                "progress": ((i + 1) / len(chunks)) * 100
            }
            yield json.dumps(error_info)
        
        # Brief pause to avoid overwhelming the API
        if i < len(chunks) - 1:
            time.sleep(1)
    
    # Create and save the improved document
    output_file = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}_improved.docx")
    improved_doc = create_improved_document(doc, paragraphs, active_jobs[job_id]["analysis_results"])
    improved_doc.save(output_file)
    
    # Notify frontend that processing is complete
    completion_info = {
        "status": "complete",
        "message": "Analysis complete!",
        "output_file": f"{job_id}_improved.docx",
        "progress": 100
    }
    yield json.dumps(completion_info)
    
    # Keep job info for download, but can clean up after a timeout
    # Could implement a cleanup job that runs periodically

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Get parameters
        model_name = request.form.get('model', DEFAULT_MODEL)
        chunk_size = int(request.form.get('chunk_size', DEFAULT_CHUNK_SIZE))
        api_url = request.form.get('api_url', DEFAULT_API_URL)
        genre = request.form.get('genre', '')
        sample_only = request.form.get('sample_only', 'false').lower() == 'true'
        
        # Create job ID and save file
        job_id = f"job_{int(time.time())}"
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(file_path)
        
        # Create job entry
        active_jobs[job_id] = {
            "file_path": file_path,
            "original_filename": filename,
            "status": "started",
            "model": model_name,
            "chunk_size": chunk_size,
            "api_url": api_url,
            "genre": genre,
            "sample_only": sample_only,
            "start_time": time.time()
        }
        
        return jsonify({"success": True, "job_id": job_id}), 200
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/process/<job_id>')
def process(job_id):
    """Stream processing results for a job."""
    if job_id not in active_jobs:
        return Response(json.dumps({"error": "Job not found"}), mimetype='application/json')
    
    job = active_jobs[job_id]
    
    def generate():
        return process_document(
            job_id, 
            job["file_path"], 
            job["model"], 
            job["chunk_size"], 
            job["api_url"],
            job["genre"],
            job["sample_only"]
        )
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/status/<job_id>')
def job_status(job_id):
    """Get the status of a job."""
    if job_id not in active_jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = active_jobs[job_id]
    total_chunks = job.get("total_chunks", 0)
    processed_chunks = job.get("processed_chunks", 0)
    
    status_info = {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "progress": (processed_chunks / total_chunks * 100) if total_chunks > 0 else 0,
        "processed_chunks": processed_chunks,
        "total_chunks": total_chunks
    }
    
    return jsonify(status_info)

@app.route('/download/<filename>')
def download_file(filename):
    """Download the improved document."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename, as_attachment=True)

@app.route('/available_models')
def available_models():
    """Get available models from Ollama."""
    api_url = request.args.get('api_url', DEFAULT_API_URL)
    
    try:
        response = requests.get(f"{api_url}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            return jsonify({"models": [model['name'] for model in models]})
        else:
            return jsonify({"error": f"Failed to get models: {response.status_code}"}), 500
    except Exception as e:
        return jsonify({"error": f"Error connecting to Ollama: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
from flask import Flask, render_template_string
import threading

app = Flask(__name__)
analysis_results_global = []  # Store analysis results globally for serving
original_paragraphs_global = []  # Store original paragraphs globally for alignment

@app.route("/")
def home():
    """Serve the analysis results as a web page."""
    if not analysis_results_global or not original_paragraphs_global:
        return "No analysis results available yet. Please run the script."
    
    # Filter out invalid results
    valid_results = [result for result in analysis_results_global if result and 'paragraphs' in result]
    if not valid_results:
        return "No valid analysis results available."

    # Flatten the analysis results to align with the original paragraphs
    detailed_analysis = []
    for chunk_result in valid_results:
        detailed_analysis.extend(chunk_result['paragraphs'])
    
    # Ensure alignment between original paragraphs and analysis results
    aligned_results = []
    for i, original_text in enumerate(original_paragraphs_global):
        if i < len(detailed_analysis):
            aligned_results.append({
                "original": original_text,
                "issues": detailed_analysis[i].get("issues", []),
                "suggestions": detailed_analysis[i].get("suggestions", []),
                "improved_version": detailed_analysis[i].get("improved_version", "")
            })
        else:
            aligned_results.append({
                "original": original_text,
                "issues": [],
                "suggestions": [],
                "improved_version": ""
            })
    
    # Render results as HTML
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Book Analysis Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            .paragraph { margin-bottom: 20px; }
            .issues, .suggestions, .improved { margin-left: 20px; }
        </style>
    </head>
    <body>
        <h1>Writing Improvement Suggestions</h1>
        <h2>Overall Assessment</h2>
        <p>{{ overall_assessment }}</p>
        <h2>Detailed Analysis</h2>
        {% for paragraph in paragraphs %}
        <div class="paragraph">
            <h3>Paragraph {{ loop.index }}</h3>
            <p><strong>Original Text:</strong> {{ paragraph.original }}</p>
            <div class="issues">
                <strong>Issues:</strong>
                <ul>
                    {% for issue in paragraph.issues %}
                    <li>{{ issue }}</li>
                    {% endfor %}
                </ul>
            </div>
            <div class="suggestions">
                <strong>Suggestions:</strong>
                <ul>
                    {% for suggestion in paragraph.suggestions %}
                    <li>{{ suggestion }}</li>
                    {% endfor %}
                </ul>
            </div>
            <div class="improved">
                <strong>Improved Version:</strong>
                <p>{{ paragraph.improved_version }}</p>
            </div>
        </div>
        {% endfor %}
    </body>
    </html>
    """
    return render_template_string(html_template, 
                                  overall_assessment=valid_results[0].get('overall_assessment', 'No assessment available'),
                                  paragraphs=aligned_results)

def run_flask():
    """Run the Flask app in a separate thread."""
    app.run(host="0.0.0.0", port=5000, debug=False)

def start_server(analysis_results, original_paragraphs):
    """Start the Flask server with the given analysis results and original paragraphs."""
    global analysis_results_global, original_paragraphs_global
    analysis_results_global = analysis_results
    original_paragraphs_global = original_paragraphs
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
"""
BookAIditor - A script that uses Ollama with Gemma 3 to analyze and improve writing quality.
This script reads a Word document, processes it through the Gemma 3 model via Ollama,
and provides writing improvement suggestions and edits.
"""

import os
import argparse
import docx
import json
import requests
import time
from typing import List
from tqdm import tqdm
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
from pydantic import BaseModel, ValidationError

class ParagraphAnalysis(BaseModel):
    original: str
    issues: List[str]
    suggestions: List[str]
    improved_version: str

class AnalysisResult(BaseModel):
    overall_assessment: str
    paragraphs: List[ParagraphAnalysis]

def parse_arguments():
    """Configure and parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze and improve the writing quality of a book')
    
    # Input/output options
    parser.add_argument('input_file', help='Path to the input document (Word or text)')
    parser.add_argument('--output', help='Path to save the output document with suggestions', default=None)
    parser.add_argument('--format', choices=['docx', 'md', 'html'], default='docx',
                       help='Output format (default: docx)')
    
    # Processing options
    parser.add_argument('--model', help='Ollama model to use', default='gemma3:1b')
    parser.add_argument('--chunk_size', type=int, help='Number of paragraphs per analysis chunk', default=3)
    parser.add_argument('--sample_only', action='store_true', help='Process only the first chunk (for testing)')
    parser.add_argument('--api_url', help='Ollama API URL', default='http://localhost:11434')
    parser.add_argument('--timeout', type=int, help='API request timeout in seconds', default=120)
    parser.add_argument('--max_retries', type=int, help='Maximum API request retries', default=3)
    parser.add_argument('--parallel', action='store_true', help='Process chunks in parallel')
    parser.add_argument('--max_workers', type=int, help='Maximum number of parallel workers', default=4)
    
    # Genre-specific flags
    genre_group = parser.add_argument_group('Genre Options')
    genre_group.add_argument('--literary_fiction', action='store_true', help='Analyze as a literary fiction novel')
    genre_group.add_argument('--thriller', action='store_true', help='Analyze as a thriller novel')
    genre_group.add_argument('--romance', action='store_true', help='Analyze as a romance novel')
    genre_group.add_argument('--sci_fi', action='store_true', help='Analyze as a science fiction novel')
    genre_group.add_argument('--young_adult', action='store_true', help='Analyze as a young adult novel')
    genre_group.add_argument('--fantasy', action='store_true', help='Analyze as a fantasy novel')
    genre_group.add_argument('--horror', action='store_true', help='Analyze as a horror novel')
    genre_group.add_argument('--mystery', action='store_true', help='Analyze as a mystery novel')
    genre_group.add_argument('--historical_fiction', action='store_true', help='Analyze as a historical fiction')
    
    # Focus-on area flags/options
    focus_group = parser.add_argument_group('Focus Options')
    focus_group.add_argument('--focus_grammar', action='store_true', help='Focus on grammar issues')
    focus_group.add_argument('--focus_style', action='store_true', help='Focus on writing style')
    focus_group.add_argument('--focus_pacing', action='store_true', help='Focus on narrative pacing')
    focus_group.add_argument('--focus_characters', action='store_true', help='Focus on character development')
    
    # Verbose mode
    parser.add_argument('--verbose', '-v', action='count', default=0, 
                       help='Increase verbosity (can be used multiple times)')
    
    # Config file
    parser.add_argument('--config', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Set output file name if not specified
    if args.output is None:
        base_name = os.path.splitext(args.input_file)[0]
        args.output = f"{base_name}_improved.{args.format}"
    
    return args

def read_word_document(file_path):
    """Read content from a Word document file."""
    print(f"Reading document: {file_path}")
    try:
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        print(f"Successfully read {len(paragraphs)} paragraphs")
        return doc, paragraphs
    except Exception as e:
        print(f"Error reading document: {e}")
        return None, []

def chunk_paragraphs(paragraphs, chunk_size):
    """Split paragraphs into chunks for processing."""
    for i in range(0, len(paragraphs), chunk_size):
        yield paragraphs[i:i + chunk_size]

def get_additional_context(args):
    """Determine additional injected prompt context"""
    if args.literary_fiction:
        return literary_fiction_context
    elif args.thriller:
        return thriller_context
    elif args.romance:
        return romance_context
    elif args.sci_fi:
        return sci_fi_context
    elif args.young_adult:
        return young_adult_context
    elif args.fantasy:
        return fantasy_context
    elif args.horror:
        return horror_context
    elif args.mystery:
        return mystery_context
    else:
        return "" # No added context

def analyze_text_with_ollama(text_chunk, model_name, api_url, additional_context=""):
    """
    Send text to Ollama API for analysis and improvement suggestions.
    Uses Pydantic to validate and parse the response.
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
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = response_text[json_start:json_end]
                    try:
                        # Use Pydantic to validate and parse
                        analysis_result = AnalysisResult.model_validate_json(json_content)
                        return analysis_result
                    except ValidationError as ve:
                        print("Pydantic validation error:", ve)
                        print("Raw JSON content:", json_content)
                        return None
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
    
    for result in analysis_results:
        if result and result.overall_assessment:
            overall_assessment = result.overall_assessment
            break
    
    new_doc.add_paragraph(overall_assessment)
    
    # Add detailed analysis for each paragraph
    new_doc.add_heading('Detailed Analysis', 1)
    
    paragraph_index = 0  # Reset paragraph index to track all paragraphs globally
    for chunk_result in analysis_results:
        if not chunk_result or not hasattr(chunk_result, "paragraphs"):
            continue
        
        for para_analysis in chunk_result.paragraphs:
            if paragraph_index >= len(paragraphs):
                break
                
            # Section for this paragraph
            new_doc.add_heading(f'Paragraph {paragraph_index + 1}', 2)
            
            # Original text
            original_heading = new_doc.add_paragraph('Original Text: ')
            original_heading.runs[0].bold = True
            new_doc.add_paragraph(paragraphs[paragraph_index])
            
            # Issues
            if para_analysis.issues:
                issues_heading = new_doc.add_paragraph('Issues: ')
                issues_heading.runs[0].bold = True
                for issue in para_analysis.issues:
                    new_doc.add_paragraph(f"• {issue}")
            
            # Suggestions
            if para_analysis.suggestions:
                suggestions_heading = new_doc.add_paragraph('Suggestions: ')
                suggestions_heading.runs[0].bold = True
                for suggestion in para_analysis.suggestions:
                    new_doc.add_paragraph(f"• {suggestion}")
            
            # Improved version
            if para_analysis.improved_version:
                improved_heading = new_doc.add_paragraph('Improved Version: ')
                improved_heading.runs[0].bold = True
                improved_para = new_doc.add_paragraph()
                improved_run = improved_para.add_run(para_analysis.improved_version)
                improved_run.font.color.theme_color = 1  # Use theme color (usually blue)
            
            new_doc.add_paragraph('---')  # Separator
            paragraph_index += 1  # Increment global paragraph index
    
    return new_doc

def main():
    # Read the document
    args = parse_arguments()
    doc, paragraphs = read_word_document(args.input_file)
    if not doc:
        print("Failed to read document. Exiting.")
        return
    
    print(f"Processing document with {args.model} model...")
    print(f"Using chunk size of {args.chunk_size} paragraphs")

    # Get additional context based on command line flags
    additional_context = get_additional_context(args)
    
    # Create chunks of paragraphs
    chunks = list(chunk_paragraphs(paragraphs, args.chunk_size))
    if args.sample_only:
        print("Sample mode: Processing only the first chunk")
        chunks = chunks[:1]
    
    # Process each chunk
    analysis_results = []
    
    for i, chunk in enumerate(tqdm(chunks, desc="Analyzing chunks")):
        chunk_text = "\n\n".join(chunk)
        print(f"\nProcessing chunk {i+1}/{len(chunks)} ({len(chunk)} paragraphs)")
        
        result = analyze_text_with_ollama(chunk_text, args.model, args.api_url, additional_context)
        if result:
            analysis_results.append(result)
        else:
            print(f"Warning: Analysis for chunk {i+1} returned None.")
        
        # Brief pause to avoid overwhelming the API
        if i < len(chunks) - 1:
            time.sleep(1)
    
    # Create and save the improved document
    if analysis_results:
        improved_doc = create_improved_document(doc, paragraphs, analysis_results)
        improved_doc.save(args.output)
        print(f"\nAnalysis complete! Improved document saved to: {args.output}")
    else:
        print("No valid analysis results available. Exiting.")

if __name__ == "__main__":
    main()
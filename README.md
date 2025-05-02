Tips for Best Results

Use --chunk_size to adjust how many paragraphs are analyzed together (smaller chunks may give more detailed analysis but take longer)

For long documents, try --sample_only first to test how the tool works on a small section
Ensure Ollama is running before executing the script

For better results, you can experiment with larger models if your system can handle it

How to run a sample: python main.py your_book.docx --sample_only
Typical operation: python main.py your_book.docx --output improved_book.docx --model gemma:3b --chunk_size 3
Or with UV: uv run main.py .\data\BookName.docx --output improved_book.docx --model gemma3:4b --chunk_size 5 --api_url http://localhost:11434

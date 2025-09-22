# 📑 MQA Academic Qualification Verification System

A local AI-powered system that automatically verifies academic staff qualifications against MQA (Malaysian Qualifications Agency) standards using document analysis and local LLMs with Ollama.

## ✨ Features

- 🔒 **Fully Local Processing** - No data leaves your machine
- 📄 **PDF Resume Analysis** - Extract education and experience from CVs
- 🎯 **MQA Compliance Checking** - Verify against Malaysian academic standards
- 👨‍🏫 **Role-Specific Analysis** - Evaluate for Teaching, Supervision, and Principal roles
- 📊 **Detailed Reports** - Generate comprehensive qualification reports
- 🖥️ **Streamlit Interface** - User-friendly web application
- 🧪 **Testing Framework** - Includes comprehensive testing scripts

## 🏗️ Project Structure

```
mqa-verification-system/
├── app.py                      # Streamlit web interface
├── verify_with_raft.py         # Core verification logic
├── extract_chunks_t.py         # Text extraction and chunking utility
├── test_raft_with_ollama.py    # Testing script for RAFT training data
├── resumes/                    # Folder for PDF resumes
├── raft_training_dataset.jsonl # Generated training data
├── chroma_db/                  # Vector database for MQA standards
├── tests/                      # Test directory
│   └── test_samples/           # Sample files for testing
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama**
   ```bash
   # Download from https://ollama.ai or use:
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Pull Required Models**
   ```bash
   ollama pull mistral
   ollama pull nomic-embed-text  # For embeddings
   ```

### Installation & Setup

1. **Clone and Setup Environment**
   ```bash
   git clone https://github.com/wiki-girl/ollama_documents_raft.gov.git
   cd ollama_documents_raft.gov
   
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # OR
   .\venv\Scripts\activate  # Windows
   
   pip install -r requirements.txt
   ```

2. **Generate Training Data** (First time setup)
   ```bash
   python verify_with_raft.py generate
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```
   Open your browser to `http://localhost:8501`
   ```
   network browser to `http://10.7.127.168:8501`

## 📖 Usage Guide

### Core Components

**1. Main Verification System (`verify_with_raft.py`)**
```bash
# Generate training data from resumes
python verify_with_raft.py generate

# Verify a specific applicant
python verify_with_raft.py verify applicant_cv.txt "Business Studies" 7
```

**2. Text Extraction Utility (`extract_chunks_t.py`)**
```bash
# Extract text chunks from PDFs for analysis
python extract_chunks_t.py --input resumes/ --output extracted_chunks/
```

**3. RAFT Testing Script (`test_raft_with_ollama.py`)**
```bash
# Test the RAFT training data with Ollama
python test_raft_with_ollama.py --samples 5 --model mistral
```

### Web Interface Usage

1. **Upload Resumes**: Place PDF resumes in the `resumes/` folder
2. **Generate Training Data**: Run `python verify_with_raft.py generate` to create RAFT dataset
3. **Verify Applicants**: Use the Streamlit interface to upload CVs and verify qualifications
4. **Select Program & Level**: Choose the academic program and qualification level
5. **Get Results**: View detailed verification reports with role-specific analysis

## 🧪 Testing Framework

The project includes comprehensive testing utilities:

### Testing RAFT Training Data
```bash
# Test with default settings (5 samples using mistral model)
python test_raft_with_ollama.py

# Test with specific parameters
python test_raft_with_ollama.py --samples 10 --model llama2 --output test_results.json
```

### Testing Text Extraction
```bash
# Test text extraction from sample PDFs
python extract_chunks_t.py --input tests/test_samples/ --output tests/extracted_output/ --verbose
```

### Expected Test Output
```
✅ Testing RAFT training data with Ollama...
📊 Loaded 15 RAFT examples
🧪 Testing 5 samples with model: mistral
📝 Sample 1: Qualification verified successfully
📝 Sample 2: Qualification verified successfully
...
✅ All tests passed! Results saved to test_output.json
```

## 🛠️ Technology Stack

- **Ollama**: Local LLM inference with Mistral model
- **LangChain**: Document processing and chain management
- **ChromaDB**: Vector database for MQA standards retrieval
- **Streamlit**: Web interface
- **PyPDF2**: PDF text extraction
- **Sentence Transformers**: Text embeddings
- **Pytest**: Testing framework

## 📋 Supported Qualification Levels

- **Level 3**: Certificate programs
- **Level 4**: Diploma programs  
- **Level 6**: Bachelor's degree programs
- **Level 7**: Master's degree programs
- **Level 8**: Doctoral programs

## ⚠️ Troubleshooting

**Common Issues:**

1. **Ollama not running**: Ensure Ollama is installed and running
2. **Model not found**: Run `ollama pull mistral` and `ollama pull nomic-embed-text`
3. **PDF extraction issues**: Check if PDFs are text-based (not scanned images)
4. **Memory errors**: Reduce chunk size in embedding settings

**Testing Issues:**
```bash
# If tests fail, try running with verbose mode
python test_raft_with_ollama.py --verbose

# Check if training data exists
python verify_with_raft.py generate
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation
- Add test cases

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_verification.py -v
```

Built with ❤️ for academic institutions in Malaysia

*Note: This system provides advisory verification. Final qualification decisions should be made by human experts.*

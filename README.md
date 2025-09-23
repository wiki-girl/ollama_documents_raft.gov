# 📑 MQA Academic Qualification Verification System

A local AI-powered system that automatically verifies academic staff qualifications against MQA (Malaysian Qualifications Agency) standards using document analysis and local LLMs with Ollama.

## ✨ Features

  - 🔒 **Fully Local Processing** - No data leaves your machine.
  - 📄 **PDF Resume Analysis** - Extract education and experience from CVs.
  - 🎯 **MQA Compliance Checking** - Verify against Malaysian academic standards.
  - 👨‍🏫 **Role-Specific Analysis** - Evaluate for Teaching, Supervision, and Principal roles.
  - 📊 **Detailed Reports** - Generate comprehensive qualification reports.
  - 🖥️ **Streamlit Interface** - User-friendly web application.
  - 🧪 **Testing Framework** - Includes comprehensive testing scripts.

-----

## 🏗️ Project Structure

```
ollama_documents_raft.gov/
├── app.py                      # Streamlit web interface
├── verify_with_raft.py         # Core verification logic
├── extract_chunks_t.py         # Text extraction and chunking utility
├── testing script/
│   └──test_raft_with_ollama.py # Testing script for RAFT training data
├── resumes/                    # Source PDFs to generate the training dataset
├── training script/
│   └──raft_training_dataset.jsonl # Generated training data
├── chroma_db/                  # Vector database for MQA standards
│   └──populate_db.py           # Script to populate the vector database
├── tests/                      # Test directory
│   └── test_samples/           # Sample files for testing
├── .env.example                # Example environment file
├── requirements.txt            # Python dependencies
├── source data/
│   └── academic.staff.txt      # Source document for MQA standards
```

-----

## 🚀 Quick Start

### Prerequisites

1.  **Install Ollama**

    ```bash
    # Download from https://ollama.ai or use:
    curl -fsSL https://ollama.ai/install.sh | sh
    ```

2.  **Pull Required Models**

    ```bash
    ollama pull mistral
    ```

### Installation & Setup

1.  **Clone and Set Up Environment**

    ```bash
    git clone https://github.com/wiki-girl/ollama_documents_raft.gov.git
    cd ollama_documents_raft.gov

    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # OR
    .\venv\Scripts\activate   # Windows

    pip install -r requirements.txt
    ```

2.  **Configure Environment**
    Create your environment file from the example.

    ```bash
    cp .env.example .env
    ```

    *You can edit the `.env` file if needed, but the defaults are set to use the `mistral` model.*

3.  **Populate the Vector Database** (First-time setup)
    This step loads the MQA standards into the local vector database.

    ```bash
    python populate_db.py
    ```

4.  **Add Source Resumes for Training** (First-time setup)
    Place your collection of PDF resumes into the `resumes/` folder. These serve as the **source data** to generate a custom training dataset.

5.  **Generate Training Data** (First-time setup)
    This command processes the PDFs in the `resumes/` folder to create the training file.

    ```bash
    python verify_with_raft.py generate
    ```

-----

## 📖 How to Use

### 🖥️ Using the Web Interface (Streamlit)

This is the recommended method for daily use.

1.  **Launch the Application**
    From your terminal, run the following command:

    ```bash
    streamlit run app.py
    ```

2.  **Access in Browser**
    Open your web browser and navigate to `http://localhost:8501`.

3.  **Upload & Verify**
    Use the file uploader to select a CV, choose the academic program and level, and view the detailed verification report.

### ⚙️ Using the Command-Line Tools

For batch processing or integration, you can use the core components directly.

**1. Main Verification System (`verify_with_raft.py`)**

```bash
# Verify a specific applicant
python verify_with_raft.py verify "resumes/applicant_cv.pdf" "Business Studies" 7
```

**2. RAFT Testing Script (`test_raft_with_ollama.py`)**

```bash
# Test the RAFT training data with Ollama
python testing script/test_raft_with_ollama.py --samples 5
```

-----

## 📋 Supported Qualifications

### Academic Program Area

This version of the system is specifically trained and configured to verify qualifications within the **Business Studies** domain. This includes, but is not limited to:

  - Finance & Accounting
  - Marketing
  - Human Resource Management
  - International Business
  - Management
  - Entrepreneurship

### Qualification Levels (MQF)

  - **Level 3**: Certificate
  - **Level 4**: Diploma
  - **Level 6**: Bachelor's Degree
  - **Level 7**: Master's Degree
  - **Level 8**: Doctoral Degree (PhD)

-----
## 🛠️ Configuration

The application is configured using an environment file:

  - `.env.example`: A template file. Do not edit this directly.
  - `.env`: Your personal configuration file, created by copying the example. This file is ignored by Git.

By default, the system is configured to use the `mistral` model for all LLM tasks.

## 🧪 Testing Framework

The project includes comprehensive testing utilities:

### Testing RAFT Training Data

```bash
# Test with default settings (5 samples using mistral model)
python test_raft_with_ollama.py

# Test with specific parameters
python test_raft_with_ollama.py --samples 10 --output test_results.json
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
-----
## 💻 Technology Stack

  - **Ollama**: Local LLM inference with Mistral model
  - **LangChain**: Document processing and chain management
  - **ChromaDB**: Vector database for MQA standards retrieval
  - **Streamlit**: Web interface
  - **PyPDF2**: PDF text extraction
  - **Sentence Transformers**: Text embeddings
  - **Pytest**: Testing framework

-----

## ⚠️ Troubleshooting

**Common Issues:**

1.  **Ollama not running**: Ensure Ollama is installed and running.
2.  **Model not found**: Run `ollama pull mistral` and `ollama pull nomic-embed-text`.
3.  **PDF extraction issues**: Check if PDFs are text-based (not scanned images).
4.  **Configuration Error**: Ensure you have created a `.env` file from the `.env.example` template.

-----

## 🤝 Contributing

Contributions are welcome\! Feel free to:

  - Report bugs and issues
  - Suggest new features
  - Submit pull requests
  - Improve documentation
  - Add test cases

Built with ❤️ for academic institutions in Malaysia

*Note: This system provides advisory verification. Final qualification decisions should be made by human experts.*

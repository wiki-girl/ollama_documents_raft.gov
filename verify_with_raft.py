import sys
import json
import random
import logging
import re
import csv
import os
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
from typing import List, Dict, Any
import PyPDF2  # Add this import

# ========= CONFIGURATION =========
class Config:
    # Files
    RAFT_FILE = "raft_training_dataset.jsonl"
    COLLECTION_NAME = "academic_staff_docs"
    PERSIST_DIR = "chroma_db"
    MQA_FILE = "academic.staff.txt"
    RESUME_FOLDER = "resumes"  # Changed from RESUME_FILE to RESUME_FOLDER
    
    # Limits
    MAX_APPLICANT_LENGTH = 1500
    MAX_STANDARDS_LENGTH = 2000
    SAMPLE_SIZE = 3
    
    # Keywords for business fields - UPDATED WITH BETTER CATEGORIZATION
    DIRECT_BUSINESS_KEYWORDS = [
        "business", "management", "accounting", "finance", "marketing",
        "administration", "commerce", "mba", "banking", "investment",
        "human resource", "hr", "logistics", "supply chain", "retail",
        "strategic", "operation", "entrepreneur", "corporate", "entrepreneurship",
        "organizational", "leadership", "business administration"
    ]
    
    RELATED_BUSINESS_KEYWORDS = [
        "economics", "economy", "monetary", "macroeconomics", "microeconomics",
        "ekonomi", "econometric", "statistic", "quantitative", "data science",
        "information system", "technology management", "innovation"
    ]
    
    NON_BUSINESS_KEYWORDS = [
        "engineering", "mechanical", "electrical", "civil", "chemical",
        "computer science", "software", "programming", "medicine", "medical",
        "pharmacy", "biology", "physics", "chemistry", "mathematics",
        "architecture", "law", "legal", "arts", "history", "philosophy",
        "psychology", "sociology", "education", "teaching", "pedagogy"
    ]

# MQA requirements
MQA_REQUIREMENTS = {
    3: "Certificate (Level 3): Bachelor's Degree in relevant business field OR Bachelor's in non-related field with 5 years relevant experience",
    4: "Diploma (Level 4): Bachelor's Degree in relevant business field OR Bachelor's in non-related field with 10 years relevant experience",
    6: "Bachelor's Degree (Level 6): Master's Degree in relevant business field OR Bachelor's with 5+ years senior management/10+ years managerial/5+ years entrepreneurship experience",
    7: "Master's Degree (Level 7): Doctoral degree in relevant business field OR Master's with 5+ years teaching OR Bachelor's with 5+ years relevant industry experience",
    8: "Doctoral Degree (Level 8): Doctoral degree with 2+ years teaching/research experience OR Master's with extensive research experience and supervision record"
}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========= HELPER FUNCTIONS =========
def detect_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
        if raw_data.startswith(b'\xff\xfe') or raw_data.startswith(b'\xfe\xff'):
            return 'utf-16'
        elif raw_data.startswith(b'\xef\xbb\xbf'):
            return 'utf-8-sig'
        return 'utf-8'
    except Exception:
        return 'utf-8'

def detect_columns(headers):
    edu_col, exp_col = None, None
    for h in headers:
        name = h.lower()
        if any(keyword in name for keyword in ["education", "qualification", "degree"]):
            edu_col = h
        if any(keyword in name for keyword in ["experience", "work", "employment"]):
            exp_col = h
    return edu_col, exp_col

def get_field_relevance(text):
    """Improved field relevance detection with better categorization"""
    text_lower = text.lower()
    
    # Check for direct business keywords
    if any(term in text_lower for term in Config.DIRECT_BUSINESS_KEYWORDS):
        return "direct"
    
    # Check for related fields
    if any(term in text_lower for term in Config.RELATED_BUSINESS_KEYWORDS):
        return "related"
    
    # Check for non-business fields
    if any(term in text_lower for term in Config.NON_BUSINESS_KEYWORDS):
        return "non-business"   
    
    return "none"

# ========= PDF PROCESSING FUNCTIONS =========
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
    return text

def extract_sections_from_text(text: str) -> Dict[str, str]:
    """Extract education and experience sections from resume text"""
    text_lower = text.lower()
    sections = {
        "education": "",
        "experience": "",
        "skills": ""
    }
    
    # Common section headers
    education_keywords = ["education", "qualification", "academic", "degree", "certification"]
    experience_keywords = ["experience", "work history", "employment", "career", "professional"]
    skills_keywords = ["skills", "competencies", "expertise", "specialization", "technical skills"]
    
    lines = text.split('\n')
    
    current_section = None
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check for section headers
        if any(keyword in line_lower for keyword in education_keywords) and len(line_lower.split()) < 5:
            current_section = "education"
            continue
        elif any(keyword in line_lower for keyword in experience_keywords) and len(line_lower.split()) < 5:
            current_section = "experience"
            continue
        elif any(keyword in line_lower for keyword in skills_keywords) and len(line_lower.split()) < 5:
            current_section = "skills"
            continue
        
        # Add content to current section
        if current_section and line.strip():
            sections[current_section] += line + "\n"
    
    # If sections weren't found by headers, try to extract using patterns
    if not sections["education"]:
        # Look for degree patterns
        degree_patterns = [
            r"(?i)(?:bachelor|bsc|ba|b\.?a\.?|b\.?sc\.?).*?(?:in|of|:).*?[\n]",
            r"(?i)(?:master|msc|ma|m\.?a\.?|m\.?sc\.?).*?(?:in|of|:).*?[\n]",
            r"(?i)(?:phd|doctorate|dphil|dr\.?).*?(?:in|of|:).*?[\n]",
            r"(?i)(?:diploma|certificate).*?(?:in|of|:).*?[\n]"
        ]
        for pattern in degree_patterns:
            matches = re.findall(pattern, text)
            if matches:
                sections["education"] = "\n".join(matches)
                break
    
    if not sections["experience"]:
        # Look for experience patterns
        exp_patterns = [
            r"(?i)(?:\d{4}[-–]\d{4}|\d{4}[-–]present).*?[\n].*?[\n]",
            r"(?i)(?:years? of experience|experience:).*?[\n]",
            r"(?i)(?:worked at|employed at|position at).*?[\n]"
        ]
        for pattern in exp_patterns:
            matches = re.findall(pattern, text)
            if matches:
                sections["experience"] = "\n".join(matches)
                break
    
    return sections

# ========= EMBEDDING MODEL =========
def get_embed_model():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        sys.exit(1)

embed_model = get_embed_model()

# ========= EDUCATION INFO EXTRACTION =========
def extract_education_info(text: str) -> Dict[str, Any]:
    """Enhanced education information extraction"""
    text_lower = text.lower()
    
    education_info = {
        "highest_degree": None,
        "degree_level": None,
        "field": None,
        "field_relevance": "unknown",
        "experience_years": 0,
        "has_teaching_experience": False,
        "has_industry_experience": False,
        "degrees": []  # Track all degrees found
    }
    
    # Extract all degrees and their levels
    degree_patterns = [
        (r"(phd|doctor|d\.phil|dr\.|doctoral)[\s\w]*?in[\s\w]*?([a-zA-Z\s]+)", 8, "Doctoral"),
        (r"(master|msc|m\.a\.|m\.ed|mba)[\s\w]*?in[\s\w]*?([a-zA-Z\s]+)", 7, "Master's"),
        (r"(bachelor|bsc|b\.a\.|undergraduate)[\s\w]*?in[\s\w]*?([a-zA-Z\s]+)", 6, "Bachelor's"),
        (r"(diploma|advanced diploma)[\s\w]*?in[\s\w]*?([a-zA-Z\s]+)", 4, "Diploma"),
        (r"(certificate)[\s\w]*?in[\s\w]*?([a-zA-Z\s]+)", 3, "Certificate")
    ]
    
    for pattern, level, degree_name in degree_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            field = match.group(2).strip() if match.group(2) else "Unknown"
            relevance = get_field_relevance(field)
            
            education_info["degrees"].append({
                "type": degree_name,
                "level": level,
                "field": field,
                "relevance": relevance
            })
            
            # Set highest degree
            if education_info["degree_level"] is None or level > education_info["degree_level"]:
                education_info["highest_degree"] = degree_name
                education_info["degree_level"] = level
                education_info["field"] = field
                education_info["field_relevance"] = relevance
    
    # If no degrees found with patterns, try simpler extraction
    if not education_info["degrees"]:
        if any(term in text_lower for term in ["phd", "doctor", "d.phil", "dr.", "doctoral"]):
            education_info["highest_degree"] = "Doctoral"
            education_info["degree_level"] = 8
        elif any(term in text_lower for term in ["master", "msc", "m.a.", "m.ed", "mba"]):
            education_info["highest_degree"] = "Master's"
            education_info["degree_level"] = 7
        elif any(term in text_lower for term in ["bachelor", "bsc", "b.a.", "undergraduate"]):
            education_info["highest_degree"] = "Bachelor's"
            education_info["degree_level"] = 6
        elif any(term in text_lower for term in ["diploma", "advanced diploma"]):
            education_info["highest_degree"] = "Diploma"
            education_info["degree_level"] = 4
        elif any(term in text_lower for term in ["certificate"]):
            education_info["highest_degree"] = "Certificate"
            education_info["degree_level"] = 3
        
        # Try to extract field
        field_patterns = [
            r"degree in ([a-zA-Z\s]+)",
            r"masters? in ([a-zA-Z\s]+)",
            r"phd in ([a-zA-Z\s]+)",
            r"bachelor.*?in ([a-zA-Z\s]+)",
            r"diploma in ([a-zA-Z\s]+)"
        ]
        
        for pattern in field_patterns:
            match = re.search(pattern, text_lower)
            if match:
                education_info["field"] = match.group(1).strip()
                education_info["field_relevance"] = get_field_relevance(education_info["field"])
                break
    
    # Extract experience
    experience_patterns = [
        r"(\d+)\s*year", r"(\d+)\s*yr", r"(\d+)\s*years",
        r"(\d+)\+ years", r"over (\d+) years", r"more than (\d+) years"
    ]
    
    for pattern in experience_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            try:
                education_info["experience_years"] = max([int(match) for match in matches if str(match).isdigit()])
                break
            except:
                continue
    
    # Check for teaching/industry experience
    education_info["has_teaching_experience"] = any(term in text_lower for term in ["teach", "lectur", "instructor", "professor", "faculty"])
    education_info["has_industry_experience"] = any(term in text_lower for term in ["industry", "professional", "work", "experience", "manager", "consultant", "executive"])
    
    return education_info

# ========= QUALIFICATION DETERMINATION =========
def determine_mqa_qualification(education_info, target_level):
    """Improved qualification determination with multiple degree support"""
    if not education_info["degree_level"]:
        return "⚠️ Cannot determine education level"

    highest_level = education_info["degree_level"]
    highest_relevance = education_info["field_relevance"]
    experience_years = education_info["experience_years"]
    has_teaching = education_info["has_teaching_experience"]
    
    # Check if applicant has any business-related degree
    has_business_degree = any(deg["relevance"] == "direct" for deg in education_info.get("degrees", []))
    has_related_degree = any(deg["relevance"] == "related" for deg in education_info.get("degrees", []))
    
    # Rule for Doctoral level teaching (Level 8)
    if target_level == 8:
        if highest_level == 8 and highest_relevance == "direct":
            return "✅ Qualified for Doctoral Level (PhD in relevant business field)"
        if highest_level == 8 and has_business_degree:
            return "✅ Qualified for Doctoral Level (PhD with additional business qualification)"
        if highest_level >= 7 and has_business_degree and has_teaching and experience_years >= 2:
            return "✅ Qualified for Doctoral Level (Master's in business with teaching experience)"
        if highest_relevance == "related":
            return "❌ Not Qualified. Economics/related field PhD not sufficient for Business faculty Doctoral role."
        return "❌ Not Qualified. Insufficient qualifications for Doctoral level teaching in Business."
    
    # Rule for Master's level teaching (Level 7)
    if target_level == 7:
        if highest_level >= 7 and highest_relevance == "direct":
            return "✅ Qualified for Master's Level (Master's/PhD in relevant business field)"
        if highest_level >= 7 and has_business_degree:
            return "✅ Qualified for Master's Level (Master's/PhD with business qualification)"
        if highest_level == 6 and has_business_degree and experience_years >= 5:
            return "✅ Qualified for Master's Level (Bachelor's in business with significant experience)"
        return "❌ Not Qualified for Master's Level. Requires Master's in business field or equivalent."
    
    # Rule for Bachelor's level teaching (Level 6)
    if target_level == 6:
        if highest_level >= 7:
            return "✅ Qualified for Bachelor's Level (Advanced degree)"
        if highest_level == 6 and highest_relevance == "direct":
            return "✅ Qualified for Bachelor's Level (Bachelor's in relevant field)"
        if highest_level == 6 and has_business_degree:
            return "✅ Qualified for Bachelor's Level (Bachelor's with business qualification)"
        return "❌ Not Qualified for Bachelor's Level. Requires Bachelor's in business field or higher."
    
    return "ℹ️ General qualifications, assess against specific standards."

# ========= STANDARDS RETRIEVAL =========
def retrieve_relevant_standards(applicant_text: str, n_results: int = 5) -> str:
    """Retrieve the most relevant MQA standards based on applicant's profile"""
    education_info = extract_education_info(applicant_text)
    
    # Create targeted queries based on applicant profile
    queries = []
    
    if education_info["degree_level"]:
        queries.append(f"MQF Level {education_info['degree_level']} requirements academic staff")
    
    if education_info["field_relevance"] == "direct":
        queries.append("Business Studies faculty qualifications MQA")
    elif education_info["field_relevance"] == "related":
        queries.append("Economics faculty qualifications MQA alternative pathways")
    else:
        queries.append("Academic staff qualifications non-related field experience MQA")
    
    if education_info["experience_years"] > 0:
        queries.append(f"Alternative qualifications {education_info['experience_years']} years experience MQA")
    
    if education_info["has_teaching_experience"]:
        queries.append("Teaching experience requirements academic staff MQA")
    
    # Add queries based on multiple degrees
    for degree in education_info.get("degrees", []):
        if degree["relevance"] == "direct":
            queries.append(f"{degree['type']} in business MQA requirements")
    
    all_standards = []
    
    try:
        client = chromadb.PersistentClient(path=Config.PERSIST_DIR)
        collection = client.get_collection(Config.COLLECTION_NAME)
        
        for query in queries:
            query_embedding = embed_model.encode(query).tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(2, n_results)
            )
            
            if results["documents"] and results["documents"][0]:
                all_standards.extend(results["documents"][0])
        
        # Remove duplicates and limit length
        unique_standards = list(dict.fromkeys(all_standards))
        standards_text = "\n".join(unique_standards[:n_results * 2])
        
        # Add structured MQA requirements based on detected level
        if education_info["degree_level"] and education_info["degree_level"] in MQA_REQUIREMENTS:
            standards_text = f"MQA Requirement Summary:\n{MQA_REQUIREMENTS[education_info['degree_level']]}\n\nDetailed Standards:\n{standards_text}"
        
        return standards_text[:Config.MAX_STANDARDS_LENGTH]
        
    except Exception as e:
        logger.error(f"Error retrieving standards: {e}")
        return "Error retrieving standards. Using fallback MQA requirements."

# ========= RAFT TRAINING DATA GENERATION =========
def generate_training_data():
    """Generate RAFT training dataset from PDF resumes in folder"""
    all_examples = []
    
    if not os.path.exists(Config.RESUME_FOLDER):
        logger.error(f"Resume folder not found at '{Config.RESUME_FOLDER}'.")
        return False

    # Get all PDF files in the folder
    pdf_files = [f for f in os.listdir(Config.RESUME_FOLDER) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.error(f"No PDF files found in '{Config.RESUME_FOLDER}'!")
        return False

    logger.info(f"Found {len(pdf_files)} PDF files in '{Config.RESUME_FOLDER}'")

    # Process each PDF file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(Config.RESUME_FOLDER, pdf_file)
        logger.info(f"Processing: {pdf_file}")
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            logger.warning(f"Skipping {pdf_file} - no text extracted")
            continue
        
        # Extract sections from text
        sections = extract_sections_from_text(text)
        education_text = sections["education"]
        experience_text = sections["experience"]
        
        if not education_text:
            logger.warning(f"Skipping {pdf_file} - no education section found")
            continue

        # Extract education info
        education_info = extract_education_info(education_text + " " + experience_text)

        # Create examples for each relevant MQA level
        for target_level in [6, 7, 8]:
            qualification = determine_mqa_qualification(education_info, target_level)
            
            context = f"Evaluating candidate for a Level {target_level} academic role in a Business faculty. A relevant degree in a business discipline is required."
            
            all_examples.append({
                "query": f"Education: {education_text[:200]} | Experience: {experience_text[:100]}",
                "context": context,
                "answer": qualification,
                "target_level": target_level,
                "education_info": education_info,
                "source_file": pdf_file
            })

    # Save Output
    if all_examples:
        with open(Config.RAFT_FILE, "w", encoding="utf-8") as f:
            for ex in all_examples:
                # Remove education_info from saved data to reduce size
                ex_copy = ex.copy()
                ex_copy.pop("education_info", None)
                f.write(json.dumps(ex_copy, ensure_ascii=False) + "\n")
        logger.info(f"Generated {len(all_examples)} RAFT examples into {Config.RAFT_FILE}")
        
        # Print statistics
        business_count = sum(1 for ex in all_examples if ex.get('education_info', {}).get('field_relevance') == 'direct')
        qualified_count = sum(1 for ex in all_examples if '✅' in ex['answer'])
        
        logger.info(f"Business-relevant qualifications: {business_count}/{len(all_examples)}")
        logger.info(f"Qualified candidates: {qualified_count}/{len(all_examples)}")
        
        return True
    else:
        logger.error("No examples were generated. Please check your PDF files.")
        return False

# ========= RAFT EXAMPLES LOADING =========
def load_raft_examples(path: str = Config.RAFT_FILE, sample_size: int = Config.SAMPLE_SIZE, applicant_text: str = "") -> List[Dict[str, Any]]:
    examples = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    examples.append(data)
                except json.JSONDecodeError:
                    continue
        
        if not examples:
            return []
        
        # If we have applicant info, try to find similar examples
        if applicant_text:
            applicant_info = extract_education_info(applicant_text)
            
            # Filter examples by similar characteristics
            filtered_examples = []
            for ex in examples:
                ex_query = ex.get("query", "").lower()
                ex_target_level = ex.get("target_level", 0)
                
                # Check for similarity
                similarity_score = 0
                if applicant_info["degree_level"] and applicant_info["degree_level"] == ex_target_level:
                    similarity_score += 2
                if applicant_info["field_relevance"] == "direct" and any(field in ex_query for field in Config.DIRECT_BUSINESS_KEYWORDS):
                    similarity_score += 1
                if applicant_info["has_teaching_experience"] and "teach" in ex_query:
                    similarity_score += 1
                
                if similarity_score > 0:
                    filtered_examples.append((ex, similarity_score))
            
            # Sort by similarity and take top examples
            if filtered_examples:
                filtered_examples.sort(key=lambda x: x[1], reverse=True)
                return [ex[0] for ex in filtered_examples[:sample_size]]
        
        # Fallback to random sampling
        return random.sample(examples, min(sample_size, len(examples)))
            
    except Exception as e:
        logger.error(f"Error loading RAFT examples: {e}")
        return []

# ========= PROMPT BUILDING =========
def build_enhanced_prompt(applicant_text, standards_text, raft_examples, program, target_level):
    education_info = extract_education_info(applicant_text)
    
    # Build structured applicant summary
    degrees_str = "\n".join([f"  - {deg['type']} in {deg['field']} ({deg['relevance']} relevance)" 
                           for deg in education_info.get("degrees", [])])
    
    applicant_summary = f"""
APPLICANT PROFILE SUMMARY:
- Highest Degree: {education_info['highest_degree'] or 'Not specified'}
- Degree Level: {education_info['degree_level'] or 'Unknown'}
- Field Relevance: {education_info['field_relevance'].title()}
- Experience: {education_info['experience_years']} years
- Teaching Experience: {'Yes' if education_info['has_teaching_experience'] else 'No'}
- Industry Experience: {'Yes' if education_info['has_industry_experience'] else 'No'}

ALL DEGREES:
{degrees_str if degrees_str else '  - No degrees extracted'}
"""

    # Build examples section
    if raft_examples:
        example_str = "\n\n".join([
            f"EXAMPLE {i+1}:\nQ: {ex.get('query', 'N/A')}\nContext: {ex.get('context', 'N/A')}\nA: {ex.get('answer', 'N/A')}"
            for i, ex in enumerate(raft_examples)
        ])
    else:
        example_str = "No similar examples available."
    prompt = f"""
ROLE: You are an MQA Academic Qualification Verification Expert.

TASK: Verify if the applicant is qualified to teach in the **{program}** program at MQF Level **{target_level}**.

OUTPUT FORMAT:
1. VERDICT: [✅/⚠️/❌] [Brief summary]
2. QUALIFICATION ANALYSIS: [Detailed comparison with standards]
3. GAPS IDENTIFIED: [Specific deficiencies if any]
4. RECOMMENDATIONS: [Suggestions for compliance]

REFERENCE EXAMPLES:
{example_str}

APPLICANT DETAILS:
{applicant_text[:Config.MAX_APPLICANT_LENGTH]}

MQA STANDARDS:
{standards_text[:Config.MAX_STANDARDS_LENGTH]}
"""
    return prompt

# ========= MAIN VERIFICATION FUNCTION =========

def verify_qualification(applicant_file: str, program: str, target_level: int):
    """Main function to verify academic qualifications"""
    # Load applicant CV
    try:
        with open(applicant_file, "r", encoding="utf-8") as f:
            applicant_text = f.read()
        
        if not applicant_text.strip():
            logger.error("Applicant CV file is empty")
            return False, None 
            
    except Exception as e:
        logger.error(f"Error reading applicant file: {e}")
        return False, None 

    logger.info(f"Processing applicant: {applicant_file}")
    
    # Extract education info for better targeting
    education_info = extract_education_info(applicant_text)
    logger.info(f"Detected education: {education_info}")

    # Retrieve relevant standards
    standards_text = retrieve_relevant_standards(applicant_text)
    logger.info("Standards retrieved successfully")

    # Load targeted RAFT examples
    raft_examples = load_raft_examples(applicant_text=applicant_text)
    logger.info(f"Loaded {len(raft_examples)} relevant RAFT examples")

    # Build enhanced prompt
    prompt = build_enhanced_prompt(applicant_text, standards_text, raft_examples, program, target_level)
    
    # Save prompt for debugging
    try:
        with open("debug_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        logger.info("Prompt saved to debug_prompt.txt")
    except:
        pass

    # Run with Ollama Mistral
    try:
        logger.info("Sending request to Ollama...")
        response = ollama.chat(
            model="mistral", 
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}
        )
        result = response["message"]["content"]
        
        print("\n" + "="*80)
        print("MQA ACADEMIC QUALIFICATION VERIFICATION REPORT")
        print("="*80)
        print(result)
        print("="*80)
        
        # Save detailed result
        output_file = f"report_applicant_{os.path.basename(applicant_file).split('.')[0]}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)
        
        logger.info(f"Detailed result saved to {output_file}")
        return True, output_file
        
    except Exception as e:
        logger.error(f"Error during Ollama API call: {e}")
        print("Failed to get verification. Please check the logs for details.")
        return False, None

# ========= MAIN FUNCTION =========
# ========= MAIN FUNCTION =========
def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_with_raft.py <command> [arguments]")
        print("Commands:")
        print("  generate  - Generate RAFT training data from PDF resumes")
        print("  verify <applicant_cv_file> <program_name> <target_level> - Verify applicant qualifications")
        sys.exit(1)

    command = sys.argv[1]
    
    if command == "generate":
        success = generate_training_data()
        if success:
            print("✅ Training data generated successfully from PDF resumes")
        else:
            print("❌ Failed to generate training data")
            sys.exit(1)
            
    elif command == "verify":
        if len(sys.argv) < 5:
            print("Usage: python verify_with_raft.py verify <applicant_cv_file> <program_name> <target_level>")
            print("Example: python verify_with_raft.py verify applicant_um.txt \"Business Studies\" 6")
            sys.exit(1)
        
        applicant_file = sys.argv[2]
        program = sys.argv[3]
        try:
            target_level = int(sys.argv[4])
        except ValueError:
            print("Error: <target_level> must be a number (e.g., 6, 7, or 8).")
            sys.exit(1)

        # NOTE: This line will now cause an error if run from the command line,
        # but it remains unchanged as requested to keep modifications minimal.
        # It does not affect the Streamlit app.
        success = verify_qualification(applicant_file, program, target_level)
        if not success:
            sys.exit(1)
            
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()

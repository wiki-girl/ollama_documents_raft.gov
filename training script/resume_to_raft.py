import json
import os
import re
import PyPDF2
from typing import List, Dict, Any

# ========= CONFIG =========
RESUME_FOLDER = "resumes"          # Folder containing PDF resumes
MQA_FILE = "academic.staff.txt"   # your standard reference
OUTPUT_FILE = "raft_training_dataset.jsonl"
# ==========================

# Load the MQA standards text
with open(MQA_FILE, "r", encoding="utf-8") as f:
    mqa_text = f.read()

# Extract MQF level requirements from the standards document
def extract_mqa_requirements(text):
    requirements = {}
    
    # Extract requirements for each MQF level
    levels = re.findall(r'MQF LEVEL: (.*?)(?=MQF LEVEL:|REMARK:|$)', text, re.DOTALL)
    
    for level_text in levels:
        level_match = re.search(r'([A-Z\s]+)\(LEVEL (\d+)\)', level_text)
        if level_match:
            level_name = level_match.group(1).strip()
            level_num = int(level_match.group(2))
            
            # Extract requirements
            req_match = re.search(r'REQUIREMENT:(.*?)(?=REMARK:|OR|$)', level_text, re.DOTALL)
            if req_match:
                requirements[level_num] = {
                    'name': level_name,
                    'requirements': req_match.group(1).strip()
                }
    
    return requirements

# Extract business-specific requirements
mqa_requirements = extract_mqa_requirements(mqa_text)

# Business Studies specific requirements based on the MQA document
BUSINESS_FACULTY_REQUIREMENTS = {
    3: {  # Certificate (Level 3)
        "minimum_degree": "Bachelor's in relevant business field",
        "alternative": "Bachelor's in non-related field with 5 years relevant experience",
        "staff_ratio": "At least 60% full-time staff",
        "min_staff": 4
    },
    4: {  # Diploma (Level 4)
        "minimum_degree": "Bachelor's in relevant business field",
        "alternative": "Bachelor's in non-related field with 10 years relevant experience",
        "staff_ratio": "At least 60% full-time staff",
        "min_staff": 6
    },
    6: {  # Bachelor's Degree (Level 6)
        "minimum_degree": "Master's in relevant business field",
        "alternative": "Bachelor's in relevant field with: 5+ years senior management OR 10+ years managerial experience OR 5+ years entrepreneurship with proven track record",
        "staff_ratio": "At least 60% full-time staff",
        "min_staff": 10
    },
    7: {  # Master's Degree (Level 7)
        "minimum_degree": "Doctoral degree in relevant business field",
        "alternative": "Master's in relevant field with 5+ years teaching experience OR Bachelor's with 5+ years relevant industry experience",
        "staff_ratio": "At least 60% full-time staff",
        "min_staff": 5
    },
    8: {  # Doctoral Degree (Level 8)
        "minimum_degree": "Doctoral degree in relevant business field with 2+ years teaching/research experience",
        "alternative": "Master's in relevant field with extensive research experience and supervision record",
        "staff_ratio": "At least 60% full-time staff",
        "min_staff": 10
    }
}

# Extract text from PDF file
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"❌ Error reading PDF {pdf_path}: {e}")
    return text

# Extract education and experience sections from text
def extract_sections_from_text(text: str) -> Dict[str, str]:
    """Extract education and experience sections from resume text"""
    text_lower = text.lower()
    sections = {
        "education": "",
        "experience": "",
        "skills": "",
        "expertise": ""
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
    
    # Extract expertise from skills section or entire text
    if sections["skills"]:
        sections["expertise"] = sections["skills"]
    else:
        sections["expertise"] = text[:500]  # Use first 500 chars for expertise detection
    
    return sections

# Detect area of expertise from text with business focus
def detect_expertise(text):
    text_lower = text.lower()
    expertise_areas = []
    
    # Business-related expertise terms from MQA document
    business_terms = [
        "business", "management", "accounting", "finance", "marketing", "economics",
        "administration", "commerce", "entrepreneur", "corporate", "banking",
        "investment", "human resource", "HR", "logistics", "supply chain", "retail",
        "strategic", "operation", "technology", "economy", "monetary", "macroeconomics"
    ]
    
    # Check for business expertise
    business_match = any(term in text_lower for term in business_terms)
    if business_match:
        expertise_areas.append("business")
    
    # Additional faculty areas
    faculty_terms = {
        "engineering": ["engineering", "technical", "mechanical", "electrical", "computer"],
        "education": ["education", "teaching", "pedagogy", "curriculum"],
        "computer_science": ["computer science", "programming", "software", "algorithm"]
    }
    
    for faculty, terms in faculty_terms.items():
        if any(term in text_lower for term in terms):
            expertise_areas.append(faculty)
    
    return expertise_areas if expertise_areas else ["general"]

# Extract education level from text
def extract_education_level(text):
    text_lower = text.lower()
    
    if any(term in text_lower for term in ["phd", "doctor", "d.phil", "dr.", "doctoral"]):
        return 8, "Doctoral"
    elif any(term in text_lower for term in ["master", "msc", "m.a.", "m.ed", "mba"]):
        return 7, "Master's"
    elif any(term in text_lower for term in ["bachelor", "bsc", "b.a.", "undergraduate"]):
        return 6, "Bachelor's"
    elif any(term in text_lower for term in ["diploma", "advanced diploma"]):
        return 4, "Diploma"
    elif any(term in text_lower for term in ["certificate"]):
        return 3, "Certificate"
    else:
        return None, "Unknown"

# Extract years of experience from text
def extract_experience_years(text):
    text_lower = text.lower()
    
    # Look for year patterns
    year_patterns = [
        r'(\d+)\s*year', r'(\d+)\s*yr', r'(\d+)\s*years',
        r'(\d+)\+ years', r'over (\d+) years', r'more than (\d+) years'
    ]
    
    for pattern in year_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            try:
                return max([int(match) for match in matches if match.isdigit()])
            except:
                continue
    
    # Look for specific number mentions
    numbers = re.findall(r'\b(\d+)\b', text_lower)
    for num in numbers:
        if 1 <= int(num) <= 50:  # Reasonable year range
            return int(num)
    
    return 0

# Check if degree is in relevant business field
def is_relevant_business_field(text):
    text_lower = text.lower()
    business_fields = [
        "business", "management", "accounting", "finance", "marketing", "economics",
        "administration", "commerce", "mba", "banking", "investment", "entrepreneur"
    ]
    return any(field in text_lower for field in business_fields)

# Determine qualification based on MQA standards
def determine_mqa_qualification(education_text, experience_text, expertise_areas, target_level=6):
    """Determine qualification based on MQA standards for target level (default: Bachelor's)"""
    
    education_level, degree_name = extract_education_level(education_text)
    experience_years = extract_experience_years(experience_text) if experience_text else 0
    is_business_relevant = is_relevant_business_field(education_text)
    is_business_faculty = "business" in expertise_areas
    
    if not education_level:
        return "⚠️ Cannot determine education level"
    
    # Get requirements for target level
    if target_level in BUSINESS_FACULTY_REQUIREMENTS:
        req = BUSINESS_FACULTY_REQUIREMENTS[target_level]
    else:
        return "❌ Invalid target qualification level"
    
    # Check against MQA standards
    if is_business_faculty:
        # Business faculty specific rules
        if education_level >= target_level and is_business_relevant:
            return f"✅ Qualified for Business Faculty ({degree_name} in relevant field meets requirements)"
        
        elif education_level == 6 and target_level == 6:  # Bachelor's for Bachelor's teaching
            if is_business_relevant and experience_years >= 5:
                return f"✅ Qualified for Business Faculty (Bachelor's with {experience_years}+ years experience)"
            else:
                return f"❌ Not Qualified for Business Faculty (Insufficient qualifications/experience)"
        
        elif education_level == 6 and target_level == 7:  # Bachelor's for Master's teaching
            if is_business_relevant and experience_years >= 5:
                return f"⚠️ Partially Qualified for Business Faculty (Bachelor's with experience, may be limited to practical components)"
            else:
                return f"❌ Not Qualified for Business Faculty (Master's required for most teaching)"
        
        elif education_level == 7 and target_level == 8:  # Master's for Doctoral teaching
            if is_business_relevant and experience_years >= 5:
                return f"⚠️ Conditionally Qualified for Business Faculty (Master's with experience, may require additional qualifications)"
            else:
                return f"❌ Not Qualified for Business Faculty (Doctoral degree typically required)"
    
    # General faculty rules
    if education_level >= target_level:
        return f"✅ Qualified ({degree_name} meets minimum requirements)"
    elif education_level == target_level - 1 and experience_years >= 5:
        return f"⚠️ Partially Qualified ({degree_name} with {experience_years}+ years experience may be considered)"
    else:
        return f"❌ Not Qualified (Insufficient qualifications for level {target_level})"

# Process a single PDF resume
def process_pdf_resume(pdf_path: str, target_level: int) -> Dict[str, Any]:
    """Process a single PDF resume and return qualification data"""
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        return None
    
    sections = extract_sections_from_text(text)
    education_text = sections["education"]
    experience_text = sections["experience"]
    expertise_text = sections["expertise"]
    
    if not education_text:
        return None
    
    # Detect expertise
    all_text = f"{education_text} {experience_text} {expertise_text}"
    expertise_areas = detect_expertise(all_text)
    
    # Determine qualification
    qualification = determine_mqa_qualification(education_text, experience_text, expertise_areas, target_level)
    
    # Create context based on MQA requirements
    if target_level in BUSINESS_FACULTY_REQUIREMENTS:
        req = BUSINESS_FACULTY_REQUIREMENTS[target_level]
        context = f"MQA Requirements for Level {target_level}: {req['minimum_degree']}. Alternative: {req['alternative']}. Staff ratio: {req['staff_ratio']}, Minimum staff: {req['min_staff']}"
    else:
        context = "Refer to general MQA Programme Standards for academic staff qualifications."
    
    education_level, degree_name = extract_education_level(education_text)
    experience_years = extract_experience_years(experience_text)
    
    return {
        "query": f"Education: {education_text[:200]}... | Experience: {experience_text[:100]}...",
        "context": context,
        "answer": qualification,
        "education_level": degree_name,
        "experience_years": experience_years,
        "is_business_relevant": is_relevant_business_field(education_text),
        "expertise_areas": expertise_areas,
        "target_level": target_level,
        "source_file": os.path.basename(pdf_path)
    }

# ===============================
# Main Processing
# ===============================
examples = []

# Check if resume folder exists
if not os.path.exists(RESUME_FOLDER):
    print(f"❌ Resume folder '{RESUME_FOLDER}' not found!")
    exit(1)

# Get all PDF files in the folder
pdf_files = [f for f in os.listdir(RESUME_FOLDER) if f.lower().endswith('.pdf')]

if not pdf_files:
    print(f"❌ No PDF files found in '{RESUME_FOLDER}'!")
    exit(1)

print(f"📋 Found {len(pdf_files)} PDF files in '{RESUME_FOLDER}'")

# Process each PDF file
for pdf_file in pdf_files:
    pdf_path = os.path.join(RESUME_FOLDER, pdf_file)
    print(f"📄 Processing: {pdf_file}")
    
    # Process for different target levels
    for target_level in [3, 4, 6, 7, 8]:
        result = process_pdf_resume(pdf_path, target_level)
        if result:
            examples.append(result)

# Save RAFT dataset
if examples:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(examples)} RAFT examples into {OUTPUT_FILE}")

    # Statistics
    business_count = sum(1 for ex in examples if ex.get('is_business_relevant', False))
    qualified_count = sum(1 for ex in examples if '✅' in ex['answer'])
    
    print(f"\n📊 Statistics:")
    print(f"   Total resumes processed: {len(pdf_files)}")
    print(f"   Total examples generated: {len(examples)}")
    print(f"   Business-relevant qualifications: {business_count}/{len(examples)}")
    print(f"   Qualified candidates: {qualified_count}/{len(examples)}")
    
    # Preview examples
    print("\n🔍 Sample examples:")
    for i, ex in enumerate(examples[:5]):
        print(f"\nExample {i+1} (Level {ex['target_level']} from {ex['source_file']}):")
        print(f"Query: {ex['query']}")
        print(f"Context: {ex['context'][:150]}...")
        print(f"Answer: {ex['answer']}")
        print(f"Expertise: {ex.get('expertise_areas', [])}")

else:
    print("❌ No examples were generated. Please check your PDF files.")

print(f"\n🎉 Processing complete! RAFT dataset saved to {OUTPUT_FILE}")
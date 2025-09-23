import json
import subprocess
import time

# Path to RAFT dataset
RAFT_FILE = "raft_training_dataset.jsonl"

def ask_ollama_mistral(prompt: str, model: str = "mistral") -> str:
    """Send a prompt to Ollama and return the response."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=True,
            timeout=30  # Add timeout to prevent hanging
        )
        return result.stdout.decode().strip()
    except subprocess.TimeoutExpired:
        return "[ERROR] Request timed out after 30 seconds"
    except subprocess.CalledProcessError as e:
        return f"[ERROR] Process failed: {e.stderr.decode()}"
    except Exception as e:
        return f"[ERROR] {str(e)}"

def create_structured_prompt(context: str, query: str, answer: str = "") -> str:
    """Create a well-structured prompt for academic qualification verification."""
    
    system_prompt = """You are an MQA (Malaysian Qualifications Agency) academic qualification verification expert. 
Your task is to evaluate academic staff qualifications against official MQA standards for Business Studies programs.

IMPORTANT INSTRUCTIONS:
1. Analyze the candidate's education and experience against the specific MQA requirements
2. Consider both the minimum qualification AND alternative pathways with experience
3. For Business Studies faculty, degrees must be in relevant fields (Business, Management, Accounting, Finance, Marketing, Economics)
4. Provide clear justification for your assessment
5. Use the exact classification labels: ✅ Qualified, ⚠️ Partially Qualified, or ❌ Not Qualified
6. Reference the specific MQF level requirements in your response

OUTPUT FORMAT:
- Start with your qualification verdict (✅/⚠️/❌)
- Briefly explain which requirement is met or not met
- Mention any alternative pathways that could apply
- Keep response concise but informative"""

    user_prompt = f"""
OFFICIAL MQA STANDARDS FOR ACADEMIC STAFF:
{context}

CANDIDATE PROFILE TO EVALUATE:
{query}

EXPECTED ASSESSMENT (for reference):
{answer if answer else "No expected answer provided"}

YOUR TASK: Based on the MQA standards above, evaluate this candidate's qualifications. Provide your assessment in the required format.
"""

    return f"{system_prompt}\n\n{user_prompt}"

def analyze_response_match(expected: str, actual: str) -> str:
    """Analyze how well the response matches the expected answer."""
    expected_lower = expected.lower()
    actual_lower = actual.lower()
    
    # Check for qualification category match
    categories = ["✅ qualified", "⚠️ partially qualified", "❌ not qualified"]
    expected_category = next((cat for cat in categories if cat in expected_lower), "")
    actual_category = next((cat for cat in categories if cat in actual_lower), "")
    
    if expected_category and actual_category:
        if expected_category == actual_category:
            return "✅ Category Match"
        else:
            return f"❌ Category Mismatch (Expected: {expected_category}, Got: {actual_category})"
    
    # Check for keyword matches
    business_keywords = ["business", "relevant field", "mqa", "experience", "qualification"]
    matches = sum(1 for keyword in business_keywords if keyword in actual_lower)
    
    if matches >= 3:
        return "⚠️ Partial Match (Keywords found)"
    else:
        return "❌ Poor Match"

def main():
    # Load RAFT dataset
    try:
        with open(RAFT_FILE, "r", encoding="utf-8") as f:
            examples = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"[ERROR] Could not find {RAFT_FILE}")
        print("Please run the resume_to_raft.py script first to generate the training data.")
        return
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {RAFT_FILE}: {e}")
        return

    print(f"✅ Loaded {len(examples)} RAFT examples")
    print("Testing first 5 examples with Ollama Mistral...\n")

    results = []
    
    # Test examples
    for i, ex in enumerate(examples[:5]):
        query = ex.get("query", "")
        context = ex.get("context", "")
        expected_answer = ex.get("answer", "")
        target_level = ex.get("target_level", 6)
        
        print("=" * 80)
        print(f"EXAMPLE {i+1} (MQF Level {target_level})")
        print("=" * 80)
        
        print(f"CONTEXT:\n{context}\n")
        print(f"QUERY:\n{query}\n")
        print(f"EXPECTED ANSWER:\n{expected_answer}\n")

        # Create structured prompt
        prompt = create_structured_prompt(context, query, expected_answer)
        
        print("--- OLLAMA RESPONSE ---")
        start_time = time.time()
        response = ask_ollama_mistral(prompt)
        response_time = time.time() - start_time
        
        print(response)
        print(f"\nResponse time: {response_time:.2f}s")
        
        # Analyze the match
        match_analysis = analyze_response_match(expected_answer, response)
        print(f"\nMATCH ANALYSIS: {match_analysis}")
        
        results.append({
            "example": i+1,
            "target_level": target_level,
            "response": response,
            "expected": expected_answer,
            "match_analysis": match_analysis,
            "response_time": response_time
        })
        
        print("=" * 80 + "\n")
        
        # Add small delay between requests
        time.sleep(1)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    
    category_matches = sum(1 for r in results if "✅ Category Match" in r["match_analysis"])
    partial_matches = sum(1 for r in results if "⚠️ Partial Match" in r["match_analysis"])
    mismatches = sum(1 for r in results if "❌" in r["match_analysis"] and "Category Match" not in r["match_analysis"])
    
    print(f"Category Matches: {category_matches}/{len(results)}")
    print(f"Partial Matches: {partial_matches}/{len(results)}")
    print(f"Mismatches: {mismatches}/{len(results)}")
    print(f"Average Response Time: {sum(r['response_time'] for r in results)/len(results):.2f}s")

    # Save results to file
    try:
        with open("ollama_test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved to ollama_test_results.json")
    except Exception as e:
        print(f"[ERROR] Could not save results: {e}")

if __name__ == "__main__":
    main()
import os
from config_t import config

def main():
    print("\n=== Extracting Text Chunks ===")
    
    # Verify source file exists
    if not os.path.exists(config.text_file_path):
        print(f"\n[ERROR] Source file not found at: {config.text_file_path}")
        print("Please ensure 'academic.staff.txt' exists in the MQA folder")
        return

    # Read and chunk the text
    try:
        with open(config.text_file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # Simple chunking by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < config.CHUNK_SIZE:
                current_chunk += f"{para}\n\n"
            else:
                chunks.append(current_chunk.strip())
                current_chunk = f"{para}\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # Save chunks
        with open(config.chunks_path, "w", encoding="utf-8") as f:
            f.write("\n\n---CHUNK---\n\n".join(chunks))
            
        print(f"\nSuccess! Created {len(chunks)} chunks in {config.chunks_path}")
        
    except Exception as e:
        print(f"\n[EXTRACTION ERROR] {str(e)}")

if __name__ == "__main__":
    main()
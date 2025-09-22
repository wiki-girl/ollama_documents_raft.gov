# app.py
import streamlit as st
import subprocess
import os
import uuid
import time
from PyPDF2 import PdfReader
import tempfile  # <-- IMPORT ADDED HERE

from verify_with_raft import verify_qualification, extract_text_from_pdf

st.set_page_config(page_title="MQA Academic Staff Verification", layout="wide")

st.title("📑 MQA Academic Staff Verification System")

# --- Program & Level Selection ---
program = st.text_input("Program Name (e.g., Business Studies)")

level_map = {
    "Certificate": 3,
    "Diploma": 4,
    "Bachelor": 6,
    "Master": 7,
    "PhD": 8
}
level_choice = st.selectbox("Select Highest Qualification Level", list(level_map.keys()))
target_level = level_map[level_choice]

# --- File Upload ---
st.subheader("Upload Resume for Verification")
uploaded_file = st.file_uploader("Upload a CV/Resume (.pdf or .txt)", type=["pdf", "txt"])

if uploaded_file:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # Extract text if PDF and save as a temporary .txt file
    if uploaded_file.type == "application/pdf":
        applicant_text = extract_text_from_pdf(temp_path)
        # Create a new temp file path for the .txt content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as txt_tmp_file:
            txt_tmp_file.write(applicant_text)
            applicant_file_path = txt_tmp_file.name
    else:
        applicant_file_path = temp_path

    # --- Run Verification ---
    if st.button("Run Verification"):
        if not program:
            st.warning("Please enter a Program Name before running verification.")
        else:
            with st.spinner(f"Running verification for **{program}** at MQF Level **{target_level} ({level_choice})**..."):
                success, report_path = verify_qualification(applicant_file_path, program, target_level)

                if success and report_path:
                    with open(report_path, "r", encoding="utf-8") as f:
                        report_text = f.read()

                    st.success("✅ Verification Completed")
                    st.subheader("Verification Report")
                    st.text_area("Report Output", report_text, height=400)

                    # --- Download button ---
                    st.download_button(
                        label="📥 Download Report",
                        data=report_text,
                        file_name=os.path.basename(report_path),
                        mime="text/plain"
                    )
                else:
                    st.error("❌ Verification failed. Please check the command prompt logs for more details.")

    # Clean up temporary files after use
    # Note: This is a simple cleanup. For robust applications, you might handle this differently.
    if 'applicant_file_path' in locals() and os.path.exists(applicant_file_path):
        os.remove(applicant_file_path)
    if 'temp_path' in locals() and os.path.exists(temp_path) and temp_path != applicant_file_path:
        os.remove(temp_path)
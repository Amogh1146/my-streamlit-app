import streamlit as st
import zipfile
import os
import shutil
import re
import pandas as pd
from utils.file_utils import extract_text_from_pdf, extract_text_from_docx
from model.embedder import get_embeddings
from model.rank_resume import rank_resumes

UPLOAD_DIR = "uploaded_resumes"
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("RecruitAI")

st.markdown("Upload a ZIP of resumes. We'll extract structured info and let you filter based on preferences. Matching resumes will be ranked and shown in a table.")

zip_file = st.file_uploader("Upload ZIP file of resumes (PDF/DOCX):", type="zip")

# --- Secure ZIP extraction ---
def extract_zip_securely(zip_file, extract_path):
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for member in zip_ref.infolist():
            filename = member.filename
            if filename.endswith('/') or "__MACOSX" in filename or filename.startswith('.') or "/." in filename:
                continue
            extracted_path = zip_ref.extract(member, extract_path)
            os.chmod(extracted_path, 0o777)
    return extract_path

# --- Extract fields from resume text ---
def extract_fields(text):
    name = re.search(r"Name[:\-]?\s*(.+)", text, re.IGNORECASE)
    prev_dept = re.search(r"(?<!Targeted )Department[:\-]?\s*(.+?)\n", text, re.IGNORECASE)
    prev_pos = re.search(r"(?<!Targeted )Position[:\-]?\s*(.+?)\n", text, re.IGNORECASE)
    target_dept = re.findall(r"Targeted Career Information.*?Department[:\-]?\s*(.*?)\n", text, re.DOTALL | re.IGNORECASE)
    target_pos = re.findall(r"Targeted Career Information.*?Position[:\-]?\s*(.*?)\n", text, re.DOTALL | re.IGNORECASE)
    skills_block = re.search(r"Core Skills(.*?)Projects", text, re.DOTALL | re.IGNORECASE)

    skills = []
    if skills_block:
        skills = re.findall(r":\s*(.*)", skills_block.group(1))
        skills = [s.strip() for line in skills for s in line.split(",")]

    return {
        "Name": name.group(1).strip() if name else "",
        "Previous Department": prev_dept.group(1).strip() if prev_dept else "",
        "Previous Position": prev_pos.group(1).strip() if prev_pos else "",
        "Targeted Department": target_dept[0].strip() if target_dept else "",
        "Targeted Position": target_pos[0].strip() if target_pos else "",
        "Skills": skills
    }

# --- Resume Processing ---
resumes_text = []
filenames = []
resume_info_list = []

if zip_file:
    extract_zip_securely(zip_file, UPLOAD_DIR)
    with st.spinner("Extracting resumes..."):
        for root, _, files in os.walk(UPLOAD_DIR):
            for file in files:
                path = os.path.join(root, file)
                if file.lower().endswith(".pdf"):
                    text = extract_text_from_pdf(path)
                elif file.lower().endswith(".docx"):
                    text = extract_text_from_docx(path)
                else:
                    continue

                if text:
                    resumes_text.append(text)
                    filenames.append(file)
                    fields = extract_fields(text)
                    fields["Resume"] = file
                    resume_info_list.append(fields)

    df_info = pd.DataFrame(resume_info_list)

    st.subheader("Select Your Preferred Fields")

    col1, col2 = st.columns(2)
    with col1:
        preferred_prev_dept = st.selectbox("Previous Department", options=[""] + sorted(df_info["Previous Department"].dropna().unique().tolist()))
        preferred_prev_pos = st.selectbox("Previous Position", options=[""] + sorted(df_info["Previous Position"].dropna().unique().tolist()))
    with col2:
        preferred_target_dept = st.selectbox("Targeted Department", options=[""] + sorted(df_info["Targeted Department"].dropna().unique().tolist()))
        preferred_target_pos = st.selectbox("Targeted Position", options=[""] + sorted(df_info["Targeted Position"].dropna().unique().tolist()))

    skill_pool = sorted(set(skill for row in resume_info_list for skill in row.get("Skills", [])))
    preferred_skills = st.multiselect("Select Required Skills", options=skill_pool)

# --- Process Button ---
if st.button("Process Resumes"):
    if not zip_file:
        st.warning("Please upload a ZIP of resumes.")
    else:
        try:
            with st.spinner("Scoring resumes..."):
                embeddings = get_embeddings(resumes_text)
                job_embedding = embeddings[0]
                resume_embeddings = embeddings[1:]

                ranked_resumes, additional_resumes = rank_resumes(job_embedding, resume_embeddings, filenames)

            # --- Prepare DataFrame for Top 10 Resumes ---
            top_resumes_df = pd.DataFrame(ranked_resumes)
            top_resumes_df["Relevance Score (%)"] = top_resumes_df["Relevance Score"].map(lambda x: f"{x * 100:.2f}%")
            top_resumes_df.drop(columns=["Relevance Score"], inplace=True)  # Remove original relevance score column
            top_resumes_df["Previous Department"] = df_info.set_index("Resume").loc[top_resumes_df["Resume"], "Previous Department"].values
            top_resumes_df["Previous Position"] = df_info.set_index("Resume").loc[top_resumes_df["Resume"], "Previous Position"].values
            top_resumes_df["Targeted Department"] = df_info.set_index("Resume").loc[top_resumes_df["Resume"], "Targeted Department"].values
            top_resumes_df["Targeted Position"] = df_info.set_index("Resume").loc[top_resumes_df["Resume"], "Targeted Position"].values
            top_resumes_df["Skills"] = df_info.set_index("Resume").loc[top_resumes_df["Resume"], "Skills"].values

            # --- Reset index to start from 1
            top_resumes_df.reset_index(drop=True, inplace=True)
            top_resumes_df.index += 1

            st.subheader("Top 10 Ranked Resumes Based on Preferences")
            st.dataframe(top_resumes_df, use_container_width=True)

            # --- Prepare DataFrame for Additional Resumes (50% ≤ score < 51%) ---
            if additional_resumes:
                additional_resumes_df = pd.DataFrame(additional_resumes)
                additional_resumes_df["Relevance Score (%)"] = additional_resumes_df["Relevance Score"].map(lambda x: f"{x * 100:.2f}%")
                additional_resumes_df.drop(columns=["Relevance Score"], inplace=True)  # Remove original relevance score column
                additional_resumes_df["Previous Department"] = df_info.set_index("Resume").loc[additional_resumes_df["Resume"], "Previous Department"].values
                additional_resumes_df["Previous Position"] = df_info.set_index("Resume").loc[additional_resumes_df["Resume"], "Previous Position"].values
                additional_resumes_df["Targeted Department"] = df_info.set_index("Resume").loc[additional_resumes_df["Resume"], "Targeted Department"].values
                additional_resumes_df["Targeted Position"] = df_info.set_index("Resume").loc[additional_resumes_df["Resume"], "Targeted Position"].values
                additional_resumes_df["Skills"] = df_info.set_index("Resume").loc[additional_resumes_df["Resume"], "Skills"].values

                # --- Reset index to start from 1
                additional_resumes_df.reset_index(drop=True, inplace=True)
                additional_resumes_df.index += 1

                st.subheader("Additional Resumes (50% ≤ Score < 51%)")
                st.dataframe(additional_resumes_df, use_container_width=True)

            # --- Optional CSV Download ---
            csv = top_resumes_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Ranked Table as CSV", data=csv, file_name="ranked_resumes.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error: {str(e)}")

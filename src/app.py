# src/app.py
import streamlit as st
import requests
import tempfile

st.set_page_config(page_title="Resume Ranker Demo", layout="wide")
st.title("AI Resume Ranker — Demo")

st.markdown("Upload a Job Description and multiple resumes (pdf/docx/txt).")

jd = st.text_area("Paste Job Description here", height=200)
uploaded_files = st.file_uploader("Upload resumes (multiple)", accept_multiple_files=True, type=['pdf','docx','txt'])

if st.button("Rank Resumes"):
    if not jd.strip():
        st.warning("Please paste the job description.")
    elif not uploaded_files:
        st.warning("Upload at least one resume.")
    else:
        with st.spinner("Ranking..."):
            files = []
            for f in uploaded_files:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f.name)
                tmp.write(f.read())
                tmp.flush()
                files.append(("files", open(tmp.name,'rb')))

            data = {"job_description": jd}
            # adjust API endpoint if running separately
            resp = requests.post("http://localhost:8000/rank", data=data, files=files)
            if resp.status_code == 200:
                results = resp.json()["results"]
                for r in results:
                    st.subheader(f"{r['filename']} — Score: {r['final_score']:.4f}")
                    st.write(f"Embed score: {r['embed_score']:.4f} | Classifier score: {r['clf_score']:.4f}")
            else:
                st.error("Error from server: " + resp.text)

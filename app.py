import time
import streamlit as st

from rag_pipeline import ask_question

st.set_page_config(
    page_title="Technical Support AI Bot",
    layout="wide"
)

st.title("Technical Support AI Bot")
st.caption("Powered by DeepInfra + FAISS + LangChain")

query = st.text_input(
    "Ask a question from the API documentation"
)

if st.button("Submit"):

    if query.strip():

        with st.spinner("Generating answer..."):

            start = time.time()

            answer, docs, latency = ask_question(query)

            latency = round(time.time() - start, 2)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Latency")
            st.write(f"{latency} seconds")

            st.subheader("Retrieved Context")

            for i, doc in enumerate(docs, start=1):

                with st.expander(f"Chunk {i}"):

                    st.write(doc.page_content)
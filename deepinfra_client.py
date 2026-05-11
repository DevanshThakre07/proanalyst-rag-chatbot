import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# load_dotenv(".env")
# # print(os.getenv("DEEPINFRA_API_KEY"))

# client = OpenAI(
#     api_key=os.getenv("DEEPINFRA_API_KEY"),
#     base_url="https://api.deepinfra.com/v1/openai"
# )

# MODEL_NAME = os.getenv("DEEPINFRA_MODEL")
load_dotenv()

API_KEY = os.getenv("DEEPINFRA_API_KEY") or st.secrets["DEEPINFRA_API_KEY"]

MODEL_NAME = (
    os.getenv("DEEPINFRA_MODEL")
    or st.secrets["DEEPINFRA_MODEL"]
)

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepinfra.com/v1/openai"
)

SYSTEM_PROMPT = """
You are a technical support assistant.

Answer the user's question ONLY using the provided context.

Rules:
1. If the answer exists in the context, provide a concise factual answer.
2. Do not use outside knowledge.
3. If the answer is not present in the context, say:
   "The provided documentation does not contain this information."
"""

def generate_llm_response(prompt):

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=300
    )

    return response.choices[0].message.content
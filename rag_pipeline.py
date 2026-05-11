from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from deepinfra_client import generate_llm_response

# Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS
vectorstore = FAISS.load_local(
    "vector_store/faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
search_kwargs={"k":5}
)

def ask_question(user_query):
    import time
    start = time.time()
    
    docs = retriever.invoke(user_query)

    context = "\n\n".join([
        doc.page_content for doc in docs
    ])

    final_prompt = f"""
You must answer ONLY from the context below.

Instructions:
- Give clear human-readable answers.
- If time is mentioned in seconds, also convert it into hours/days when appropriate.
- Keep answers concise and factual.

Context:
{context}

User Question:
{user_query}

Answer:
"""

    answer = generate_llm_response(final_prompt)
    latency = round(time.time() - start, 2)
    return answer, docs, latency

if __name__ == "__main__":

    while True:

        query = input("\nAsk Question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        answer, docs = ask_question(query)

        print("\n" + "=" * 50)
        print("ANSWER:\n")
        print(answer)

        print("\n" + "=" * 50)
        print("RETRIEVED CHUNKS:\n")

        for i, doc in enumerate(docs, start=1):

            print(f"\n--- Chunk {i} ---\n")
            print(doc.page_content[:1000])
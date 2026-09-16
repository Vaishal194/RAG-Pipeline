from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
persistent_directory="db/chroma_db"
embedding_model=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
model=GoogleGenerativeAI(model="gemini-2.5-flash",temperature=0)
query="Write a program to reverse a linked list"

db=Chroma(
    collection_metadata={"hnsw:space":"cosine"},
    embedding_function=embedding_model,
    persist_directory=persistent_directory
)
retriever=db.as_retriever(search_kwargs={"k":5})
rel_doc=retriever.invoke(query)

context="\n\n".join(doc.page_content for doc in rel_doc)

answer=f'''Hey gemini,You are an Expert RAG System that accepts the chunks or the relevant queries and contexts and provides responses based
on relevant to query provided.
*If the Query is Out of Context, Answer as "I Dont have enough Information about the query"
Query: {query}
Context: {context}
Keep the reponse clear,concise and understandable to the user and dont provide answers out of context
'''
response=model.invoke(answer)
print(response)
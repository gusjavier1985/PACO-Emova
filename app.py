import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

st.title("🤖 Asistente PACO - Línea B")
st.write("Consulta normativa técnica")

Cargar base de datos
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
Nota: Para que esto funcione en la nube, la carpeta 'base_conocimiento_paco' debe estar subida a este mismo repositorio
db = FAISS.load_local("base_conocimiento_paco", embeddings, allow_dangerous_deserialization=True)

Configurar IA
llm = ChatGroq(api_key="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", model_name="llama-3.3-70b-versatile")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

pregunta = st.text_input("Tu consulta:")
if pregunta:
respuesta = qa.run(pregunta)
st.write(respuesta)


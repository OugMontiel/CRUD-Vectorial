import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, GenerationConfig
import torch
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
import shutil
from functools import lru_cache


# Configuración inicial
# MODEL_NAME = "google/flan-t5-base"  # Versión balanceada
# Alternativas:
MODEL_NAME = "google/flan-t5-small"  # Versión más ligera
# MODEL_NAME = "google/flan-t5-large"  # Si tienes más recursos
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Más ligero
PERSIST_DIRECTORY = "chroma_db"

# Función para cargar documentos
def load_documents(uploaded_files):
    documents = []
    for file in uploaded_files:
        file_path = os.path.join("temp", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        try:
            if file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.lower().endswith(".txt"):
                loader = TextLoader(file_path)
            elif file.name.lower().endswith(".csv"):
                loader = CSVLoader(file_path)
            elif file.name.lower().endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file.name.lower().endswith(".html") or file.name.lower().endswith(".latex"):
                loader = UnstructuredHTMLLoader(file_path)
            else:
                continue
            
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error al cargar {file.name}: {str(e)}")
    
    return documents

# Función para entrenar el modelo
def train_model(documents, chunk_size, chunk_overlap, preprocess):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    processed_docs = text_splitter.split_documents(documents)
    
    # Asegurar que usamos el modelo correcto de embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    
    # Crear nueva base de datos con dimensión consistente
    vectordb = Chroma.from_documents(
        documents=processed_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    vectordb.persist()
    
    return vectordb

# Preprocesamiento de texto
def preprocess_text(text):
    text = text.lower()
    # Aquí se pueden añadir más pasos de preprocesamiento
    return text

# Configuración de la interfaz
st.set_page_config(page_title="DeepSeek Interface", layout="wide")

# Inicialización del estado de la sesión
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

# Barra lateral para carga de documentos y configuración
with st.sidebar:
    st.header("Configuración")
    
    # Carga de documentos
    uploaded_files = st.file_uploader(
        "Cargar documentos (PDF, TXT, CSV, DOCX, LaTeX)",
        type=["pdf", "txt", "csv", "docx", "latex", "html"],
        accept_multiple_files=True
    )
    
    # Parámetros de entrenamiento
    st.subheader("Parámetros de Entrenamiento")
    chunk_size = st.slider("Tamaño de fragmentos", 256, 2048, 512)
    chunk_overlap = st.slider("Solapamiento de fragmentos", 0, 256, 128)
    preprocess = st.checkbox("Preprocesar texto (limpieza básica)")
    
    # Borrar la base de datos existente:
    shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)  # Ejecutar esto antes de entrenar (train_model)

    if st.button("Entrenar Modelo") and uploaded_files:
        with st.spinner("Procesando documentos..."):
            os.makedirs("temp", exist_ok=True)
            documents = load_documents(uploaded_files)
            st.session_state.vector_db = train_model(
                documents, chunk_size, chunk_overlap, preprocess
            )
            st.success("Modelo entrenado con éxito!")

# Sección principal para chats
st.header("Chat con DeepSeek")

# Gestión de chats
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Nuevo Chat"):
        chat_id = f"Chat {len(st.session_state.chats)+1}"
        st.session_state.chats[chat_id] = []
        st.session_state.current_chat = chat_id

    selected_chat = st.selectbox(
        "Chats existentes",
        options=list(st.session_state.chats.keys()),
        index=len(st.session_state.chats)-1 if st.session_state.chats else 0
    )
    st.session_state.current_chat = selected_chat

# Historial del chat
chat_container = st.container()

# Entrada de usuario
user_input = st.chat_input("Escribe tu pregunta...")

# Gestión de Caché para Modelos:
@lru_cache(maxsize=1)
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

@lru_cache(maxsize=1)
def load_llm_model():
    return AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# generación de respuestas:
def generate_response(prompt, context):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_llm_model()
    
    input_text = f"Contexto: {context}\nPregunta: {prompt}\nRespuesta:"
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=384,  # Reducir máximo para ahorrar memoria
        truncation=True
    )
    
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=150,  # Reducir longitud de respuesta
        temperature=0.65,
        repetition_penalty=1.25,
        num_beams=2  # Menor consumo de memoria
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if user_input and st.session_state.vector_db:
    # Búsqueda de documentos relevantes
    docs = st.session_state.vector_db.similarity_search(user_input, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Generación de respuesta
    try:
        response = generate_response(user_input, context)
        
        # Actualización del historial del chat
        if st.session_state.current_chat:
            st.session_state.chats[st.session_state.current_chat].append(
                {"user": user_input, "bot": response}
            )
            
    except Exception as e:
        st.error(f"Error generando respuesta: {str(e)}")
    
    # Actualización del historial del chat
    if st.session_state.current_chat:
        st.session_state.chats[st.session_state.current_chat].append(
            {"user": user_input, "bot": response[0]['generated_text']}
        )

# Mostrar historial del chat
with chat_container:
    if st.session_state.current_chat:
        for message in st.session_state.chats[st.session_state.current_chat]:
            with st.chat_message("user"):
                st.write(message["user"])
            with st.chat_message("assistant"):
                st.write(message["bot"])

# Guardar conversaciones
if st.session_state.chats and st.button("Guardar Conversación"):
    chat_history = "\n\n".join(
        [f"Usuario: {msg['user']}\nDeepSeek: {msg['bot']}" 
         for msg in st.session_state.chats[st.session_state.current_chat]]
    )
    st.download_button(
        label="Descargar Conversación",
        data=chat_history,
        file_name="conversacion.txt",
        mime="text/plain"
    )
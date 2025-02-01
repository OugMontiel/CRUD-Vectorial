import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
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

# Configuración inicial
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
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
    
    if preprocess:
        for doc in processed_docs:
            doc.page_content = preprocess_text(doc.page_content)
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    
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

# Y en la sección de generación de respuestas, modificar:
def generate_response(prompt, context):
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True  # Usar cuantización 4-bit ... para pocos recursos
    )
    
    # Formatear el prompt según requiere Deepseek
    messages = [
        {"role": "user", "content": f"Contexto: {context}\n\nPregunta: {prompt}"}
    ]
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Configurar parámetros de generación
    generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config.max_new_tokens = 512
    generation_config.temperature = 0.7
    generation_config.top_p = 0.9
    generation_config.do_sample = True
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        generation_config=generation_config,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("assistant\n")[-1].strip()

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
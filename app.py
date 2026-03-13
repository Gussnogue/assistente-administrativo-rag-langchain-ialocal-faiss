import streamlit as st
import os
import tempfile
import numpy as np
import faiss
import pickle
from dotenv import load_dotenv
from openai import OpenAI
from markitdown import MarkItDown
from sentence_transformers import SentenceTransformer

load_dotenv()

# ===== CONFIGURAÇÃO =====
LM_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "hermes-3-llama-3.2-3b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1.5")

# ===== INICIALIZAÇÃO =====
@st.cache_resource
def init_embedder():
    # Tenta usar modelo via sentence-transformers (local) se disponível
    try:
        return SentenceTransformer(EMBEDDING_MODEL)
    except:
        # Fallback: usar API do LM Studio para embeddings
        return None

embedder = init_embedder()
client = OpenAI(base_url=LM_URL, api_key="not-needed")

# ===== FUNÇÕES =====
def pdf_para_markdown(caminho_pdf):
    """Converte PDF para Markdown usando MarkItDown da Microsoft"""
    md = MarkItDown()
    resultado = md.convert(caminho_pdf)
    return resultado.text_content

def chunk_markdown(texto, chunk_size=500, overlap=50):
    """Divide Markdown em chunks com overlap (simples)"""
    palavras = texto.split()
    chunks = []
    for i in range(0, len(palavras), chunk_size - overlap):
        chunk = " ".join(palavras[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def gerar_embedding(texto):
    """Gera embedding via sentence-transformers (se disponível) ou via API"""
    if embedder:
        return embedder.encode([texto])[0].tolist()
    else:
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texto
        )
        return resp.data[0].embedding

# ===== INTERFACE STREAMLIT =====
st.set_page_config(page_title="Assistente Administrativo", layout="wide")
st.title("📄 Assistente Administrativo com IA Local")
st.markdown("Envie um PDF e faça perguntas sobre seu conteúdo. Tudo roda localmente.")

# Sidebar com status
with st.sidebar:
    st.header("🔧 Status")
    st.success(f"Modelo Chat: {CHAT_MODEL}")
    st.success(f"Modelo Embedding: {EMBEDDING_MODEL}")
    
    # Botão para resetar
    if st.button("🧹 Resetar conversa"):
        st.session_state.clear()
        st.rerun()

# Upload do arquivo
uploaded_file = st.file_uploader("Escolha um arquivo PDF", type=["pdf"])

if uploaded_file:
    # Processar PDF (só uma vez)
    if "chunks" not in st.session_state:
        with st.spinner("📄 Convertendo PDF para Markdown..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            markdown_text = pdf_para_markdown(tmp_path)
            os.unlink(tmp_path)
            
            st.info(f"✅ PDF convertido! {len(markdown_text)} caracteres")
            
            # Dividir em chunks
            chunks = chunk_markdown(markdown_text)
            st.success(f"📦 {len(chunks)} chunks gerados")
            
            # Gerar embeddings e criar índice FAISS
            with st.spinner("🔍 Gerando embeddings..."):
                embeddings = []
                for chunk in chunks:
                    emb = gerar_embedding(chunk)
                    embeddings.append(emb)
                
                embeddings_array = np.array(embeddings).astype('float32')
                index = faiss.IndexFlatL2(embeddings_array.shape[1])
                index.add(embeddings_array)
                
                # Salvar na sessão
                st.session_state["chunks"] = chunks
                st.session_state["index"] = index
                st.session_state["mensagens"] = []
                
                st.success("✅ Pronto! Faça perguntas abaixo.")

# Chat
if "chunks" in st.session_state:
    st.divider()
    
    # Histórico
    for msg in st.session_state["mensagens"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Input
    if prompt := st.chat_input("Faça uma pergunta sobre o documento"):
        st.session_state["mensagens"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Buscando..."):
                # Embedding da pergunta
                query_emb = gerar_embedding(prompt)
                query_array = np.array([query_emb]).astype('float32')
                
                # Busca no FAISS
                distances, indices = st.session_state["index"].search(query_array, k=3)
                
                # Montar contexto
                contexto = "\n\n".join([st.session_state["chunks"][i] for i in indices[0]])
                
                # Chamar LM Studio
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": "Você é um assistente administrativo. Responda com base no contexto fornecido."},
                        {"role": "user", "content": f"Contexto:\n{contexto}\n\nPergunta: {prompt}"}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                resposta = response.choices[0].message.content
                st.markdown(resposta)
                
                # Mostrar fontes
                with st.expander("📚 Fontes consultadas"):
                    for i, idx in enumerate(indices[0]):
                        st.write(f"**Trecho {i+1}:**")
                        st.write(st.session_state["chunks"][idx][:200] + "...")
            
            st.session_state["mensagens"].append({"role": "assistant", "content": resposta})

            
# 📄 Assistente Administrativo RAG com IA Local

Um sistema de **RAG (Retrieval-Augmented Generation)** que processa documentos PDF e responde perguntas em linguagem natural. O pipeline inclui conversão para Markdown (MarkItDown), segmentação de texto, embeddings com **nomic-embed-text-v1.5**, indexação vetorial com **FAISS** e geração de respostas via **Hermes 3 3B** (LM Studio). Tudo rodando **100% offline**, com total privacidade e controle dos seus dados.

* * *

## ✨ Funcionalidades

* ✅ **Upload de PDF:** Envie qualquer documento PDF para processamento.
* ✅ **Conversão Inteligente:** Uso da biblioteca `markitdown` da Microsoft para transformar PDF em Markdown limpo.
* ✅ **Chunking Automático:** Divisão do texto em segmentos com overlap para preservar contexto.
* ✅ **Embeddings Locais:** Geração de vetores semânticos com o modelo **nomic-embed-text-v1.5** (sentence-transformers).
* ✅ **Índice FAISS:** Busca eficiente dos trechos mais relevantes para cada pergunta.
* ✅ **Chat Contextual:** Respostas geradas pelo modelo **Hermes 3 3B** (via LM Studio) baseadas exclusivamente no documento.
* ✅ **100% Offline:** Nenhum dado sai da sua máquina, sem dependência de APIs externas ou nuvem.
* ✅ **Interface Streamlit:** Fácil de usar e totalmente customizável.

* * *

## 🚀 Como Executar

### Pré-requisitos

* **Python 3.10+**
* **LM Studio** com os seguintes modelos carregados:
  * `nomic-embed-text-v1.5` (para embeddings)
  * `hermes-3-llama-3.2-3b` (para chat)
* **Servidor do LM Studio** ativo na porta 1234

### Passo a Passo

1. **Clone o repositório**

   ```bash
   git clone https://github.com/Gussnogue/assistente-administrativo-rag-langchain-ialocal-faiss.git
   cd assistente-administrativo-rag-langchain-ialocal-faiss

2. **Crie e ative um ambiente virtual**

   ```bash
   python -m venv venv
   # Windows (CMD)
   venv\Scripts\activate.bat
   # Linux/Mac
   source venv/bin/activate

3. **Instale as dependências**

   ```bash
   pip install -r requirements.txt

4. **Configure as variáveis de ambiente**
   
   ```bash
   LM_STUDIO_URL=http://localhost:1234/v1
   MODELO_CHAT=hermes-3-llama-3.2-3b
   MODELO_EMBEDDING=nomic-embed-text-v1.5
   CHUNK_SIZE=500
   CHUNK_OVERLAP=50
   TOP_K=3

5. **Inicie o LM Studio**

   ```
   Abra o LM Studio
   Carregue os modelos: nomic-embed-text-v1.5 e hermes-3-llama-3.2-3b

7. **Execute o aplicativo**

   ```bash
   streamlit run app.py

### Licença
Este projeto está licenciado sob a MIT License. Veja o arquivo LICENSE para mais detalhes.

### 🤝 Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

Azitech-GustavoNogueira

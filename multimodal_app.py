import os
import io
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile
from PIL import Image
import google.generativeai as genai

st.set_page_config(page_title="Multimodal AI Knowledge Assistant", page_icon="🧠")
st.sidebar.title("🧠 Multimodal AI Assistant")
st.sidebar.markdown("Upload PDFs, Images, or ask any question!")

groq_api_key = os.environ.get("GROQ_API_KEY", "")
gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
if not gemini_api_key:
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")

st.title("🧠 Multimodal AI Knowledge Assistant")
st.markdown("Powered by LLaMA 3.3 70B + Gemini Vision + FAISS")

tab1, tab2, tab3 = st.tabs(["📄 Document Q&A", "🖼️ Image Analysis", "💬 General Chat"])

if groq_api_key:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=groq_api_key)

    with tab1:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "qa_chain" not in st.session_state:
            st.session_state.qa_chain = None
        if "retriever" not in st.session_state:
            st.session_state.retriever = None

        uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True, key="pdf_uploader")

        if uploaded_files:
            with st.spinner("Processing documents..."):
                all_chunks = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                        f.write(uploaded_file.read())
                        temp_path = f.name
                    loader = PyPDFLoader(temp_path)
                    documents = loader.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = splitter.split_documents(documents)
                    all_chunks.extend(chunks)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vector_store = FAISS.from_documents(all_chunks, embeddings)
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                st.session_state.retriever = retriever
                prompt = PromptTemplate.from_template("""You are a helpful AI assistant. Use the following context to answer the question accurately. If you don't know the answer from the context, say "I don't have enough information to answer this."
Context: {context}
Question: {question}
Answer:""")
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                st.session_state.qa_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt | llm | StrOutputParser()
                )
            st.success(f"✅ {len(all_chunks)} chunks indexed!")

        if st.session_state.qa_chain:
            question = st.chat_input("Ask about your documents...")
            if question:
                with st.spinner("Thinking..."):
                    answer = st.session_state.qa_chain.invoke(question)
                    source_docs = st.session_state.retriever.invoke(question)
                    sources = list(set([f"Page {doc.metadata.get('page', 0) + 1}" for doc in source_docs]))
                st.session_state.chat_history.append({"question": question, "answer": answer, "sources": sources})
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(chat["question"])
                with st.chat_message("assistant"):
                    st.write(chat["answer"])
                    st.caption(f"📚 Sources: {', '.join(chat['sources'])}")

    with tab2:
        st.markdown("### 🖼️ Image Analysis")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            vision_model = genai.GenerativeModel("gemini-1.5-flash")
            uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader")
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                img_question = st.text_input("Ask about the image")
                if img_question:
                    with st.spinner("Analyzing image..."):
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format=image.format or "PNG")
                        img_bytes = img_byte_arr.getvalue()
                        image_part = {"mime_type": f"image/{(image.format or 'PNG').lower()}", "data": img_bytes}
                        response = vision_model.generate_content([img_question, image_part])
                    st.markdown("### Answer")
                    st.write(response.text)
        else:
            st.info("Please enter your Gemini API key in the sidebar to use image analysis.")

    with tab3:
        st.markdown("### 💬 General Chat")
        if "general_history" not in st.session_state:
            st.session_state.general_history = []
        general_question = st.chat_input("Ask me anything...", key="general_chat")
        if general_question:
            with st.spinner("Thinking..."):
                response = llm.invoke(general_question)
            st.session_state.general_history.append({"question": general_question, "answer": response.content})
        for chat in st.session_state.general_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])

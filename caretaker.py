import os
import streamlit as st
from dotenv import load_dotenv
from datasets import load_dataset
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.callbacks.base import BaseCallbackHandler

# Load environment variables
load_dotenv("open_ai.env")
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# Custom CallbackHandler for streaming
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

# Sidebar sliders
chattiness = st.sidebar.slider("Chattiness Level", min_value=1, max_value=10, value=7)
max_tokens_slider = st.sidebar.slider("Max Tokens", min_value=30, max_value=300)

temperature = 0.2 + (chattiness - 1) * 0.096
max_tokens = 30 + (max_tokens_slider - 1) * 3

st.sidebar.markdown(f"🌡️ Temperature: `{temperature:.2f}`")
st.sidebar.markdown(f"✏️ Max Tokens: `{max_tokens}`")

# Load dataset and convert to documents
@st.cache_resource(show_spinner="Loading and processing data...")
def prepare_data():
    data = load_dataset("ShenLab/MentalChat16K")["train"]
    
    # Only include items with both input and output
    docs = []
    for item in data:
        input_text = item.get("input")
        output_text = item.get("output")
        if input_text and output_text:
            full_text = input_text.strip() + "\n" + output_text.strip()
            docs.append(Document(page_content=full_text))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Initialize LLM + Retrieval Chain
def initialize_chain(retriever):
    llm = ChatOpenAI(
        model_name="llama3-70b-8192",
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key,
        streaming=True
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an empathetic and emotionally intelligent AI caretaker. "
            "Your job is to deeply understand and support users who are struggling emotionally. "
            "Be warm, encouraging, and non-judgmental.\n"
        ),
        HumanMessagePromptTemplate.from_template("{context}\n\nUser: {question}")
    ])

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt, "document_variable_name": "context"}
    )

# Streamlit App
st.set_page_config(page_title="AI CareTaker 💖", page_icon="💌")
st.title("💖 Your Loving AI CareTaker")
st.write("Tell me how you're feeling...💗")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    retriever = prepare_data()
    st.session_state.qa_chain = initialize_chain(retriever)

user_input = st.chat_input("Start a conversation...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("assistant"):
        response_area = st.empty()
        stream_handler = StreamHandler(response_area)

        result = st.session_state.qa_chain.invoke(
            {"question": user_input},
            config={"callbacks": [stream_handler]}
        )

        answer = result.get("answer", result)
        st.session_state.chat_history.append(("assistant", answer))

# Show chat history
for sender, msg in st.session_state.chat_history:
    st.chat_message(sender).markdown(msg)

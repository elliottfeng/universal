import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# 设置页面标题
st.title("💬 冯宇洋のChatbot")

# 应用介绍
st.markdown("""
这是一个基于 LangChain 框架和 DeepSeek 模型的智能对话机器人。它能够理解你的问题，并从本地知识库中检索相关信息，为你提供准确的回答。

#### 主要功能：
- **智能对话**：询问关于我的一切，理解并回答你的问题。

快来试试吧！😊
""")

# 定义本地文本文件路径
TEXT_FILE_PATH = "knowledge_base.txt"  # 替换为你的本地文本文件路径

# 初始化历史记录
if "history" not in st.session_state:
    st.session_state.history = []

# 初始化主模型
MAIN_MODEL = ChatOpenAI(
    model="deepseek-v4-flash",  # 主模型名称
    api_key="sk-mcPCh2zIXjSTN53c23B73c9316D74e47A50eD42c52692a43",  # 主模型 API 密钥
    base_url="https://vip.apiyi.com/v1",  # 主模型 API 地址
    max_tokens=1024
)

# 初始化备用模型
BACKUP_MODEL = ChatOpenAI(
    model="gpt-4o-mini",  # 备用模型名称
    api_key="sk-mcPCh2zIXjSTN53c23B73c9316D74e47A50eD42c52692a43",  # 备用模型 API 密钥
    base_url="https://vip.apiyi.com/v1",
    max_tokens=1024
)

# 使用在线模型初始化 Embeddings
embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

# 加载本地文本文件并初始化向量存储
def init_vector_store(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在，请提供有效的文本文件路径。")

    # 读取文本文件
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 使用文本分割器将文本分割成小块
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)

    # 创建向量存储
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

# 初始化 RetrievalQA
def init_retrieval_qa(llm, vector_store):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    return qa

# 获取模型回复
def get_model_response(model, query, vector_store):
    # 初始化 RetrievalQA
    qa = init_retrieval_qa(model, vector_store)

    # 调用 RetrievalQA 生成回复
    response = qa.run(query)
    return response

# 获取模型回复（带超时控制）
def get_response_with_timeout(query, vector_store, timeout=10):
    with ThreadPoolExecutor() as executor:
        # 提交主模型请求
        future = executor.submit(get_model_response, MAIN_MODEL, query, vector_store)
        try:
            # 设置超时时间
            response = future.result(timeout=timeout)
            return response
        except TimeoutError:
            # 如果主模型超时，切换到备用模型
            st.warning("主模型响应超时，正在切换到备用模型...")
            return get_model_response(BACKUP_MODEL, query, vector_store)

# 显示当前对话记录
def show_current_session():
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 初始化向量存储
vector_store = init_vector_store(TEXT_FILE_PATH)

# 接收用户输入
if user_input := st.chat_input("Chat with 冯宇洋: "):
    # 将用户的输入加入历史记录
    st.session_state.history.append({"role": "user", "content": user_input})

    # 使用 spinner 提示“正在输入...请耐心等待”
    with st.spinner("正在输入...请耐心等待。"):
        # 获取模型生成的回复（带超时控制）
        response = get_response_with_timeout(user_input, vector_store)

    # 将模型的输出加入到历史记录
    st.session_state.history.append({"role": "assistant", "content": response})

# 显示当前对话记录
show_current_session()

import streamlit as st
from langchain.chains.prompt_selector import is_llm
from langchain.llms.base import LLM
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# 设置页面配置
st.set_page_config(page_title="多语种翻译", page_icon="🌐", layout="wide")

# 初始化 session_state
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False

# 在页面上标注名字和邮箱
st.sidebar.markdown("**冯宇洋**")
st.sidebar.markdown("邮箱: [elliottfeng@mail.com](mailto:elliottfeng@mail.com)")

# 侧边栏：输入 API 密钥
with st.sidebar:
    st.title("设置")
    api_key = st.text_input("请输入 API 密钥", type="password", help="请确保 API 密钥正确，否则无法调用翻译服务。", value="sk-mcPCh2zIXjSTN53c23B73c9316D74e47A50eD42c52692a43")

# 主页面
st.title("文本翻译工具")
st.markdown("请输入需要翻译的文本，点击“翻译”按钮获取结果。")

# 输入框
input_text = st.text_area("请输入文本", placeholder="例如：Hello, how are you?")

# 初始化主模型和备用模型
MAIN_MODEL = ChatOpenAI(
    model="deepseek-chat",  # 主模型名称
    api_key=api_key,  # 主模型 API 密钥
    base_url="https://vip.apiyi.com/v1",  # 主模型 API 地址
    max_tokens=1024
)

BACKUP_MODEL = ChatOpenAI(
    model="gpt-4o-mini",  # 备用模型名称
    api_key=api_key,  # 备用模型 API 密钥
    base_url="https://vip.apiyi.com/v1",  # 备用模型 API 地址
    max_tokens=1024
)

# 构建系统提示模板
system_template_text = (
    "你是一位全能的翻译，能够识别出任何语言，并且将识别出的语言翻译成中文。"
    "请确保翻译准确、流畅，并输出翻译后的文本，不要有任何其他内容。"
    "如果输入是单词或者短语（不是完整的句子），需要能够以字典的形式格式化显示输入的单词或短语的音标，读音，意思，词性以及输入词的造句（翻译）。"
)
system_prompt_template = SystemMessagePromptTemplate.from_template(system_template_text)

# 构建用户输入模板
human_template_text = "{text}"
human_prompt_template = HumanMessagePromptTemplate.from_template(human_template_text)

# 获取模型回复（带超时控制）
def get_translation_with_timeout(model, input_text, timeout=10):
    with ThreadPoolExecutor() as executor:
        # 提交模型请求
        future = executor.submit(model.invoke, [system_prompt_template.format(), human_prompt_template.format(text=input_text)])
        try:
            # 设置超时时间
            response = future.result(timeout=timeout)
            return response
        except TimeoutError:
            # 如果主模型超时，切换到备用模型
            st.warning("主模型响应超时，正在切换到备用模型...")
            return BACKUP_MODEL.invoke([system_prompt_template.format(), human_prompt_template.format(text=input_text)])

# 翻译按钮
if st.button("翻译", disabled=st.session_state.button_clicked):
    if not api_key:
        st.error("请先在侧边栏输入 API 密钥！")
    elif not input_text.strip():
        st.error("请输入需要翻译的文本！")
    else:
        # 设置按钮状态为“已点击”
        st.session_state.button_clicked = True

        # 显示加载提示
        with st.spinner("正在翻译中，请不要重复点击..."):
            try:
                # 调用主模型（带超时控制）
                response = get_translation_with_timeout(MAIN_MODEL, input_text)

                # 显示翻译结果
                st.success("翻译完成！")
                st.markdown("### 翻译结果：")

                # 将翻译结果显示在文本框中
                result_text = response.content
                result_area = st.text_area("翻译结果", value=result_text, height=300)

                # 恢复按钮状态
                st.session_state.button_clicked = False
            except Exception as e:
                st.error(f"翻译失败：{str(e)}")
                # 恢复按钮状态
                st.session_state.button_clicked = False

# 如果按钮状态为“已点击”，禁用按钮
if st.session_state.button_clicked:
    st.button("翻译", disabled=True)
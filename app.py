import streamlit as st
import os

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

from video import get_text_from_youtube_video_url
from website import get_text_from_website_url

# 使用PDF读取器，按页读取内容
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    print(text)
    return text

# 文本分块
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# 文字矢量入库
def get_vectorstore(text_chunks):
    if len(text_chunks) == 0:
        st.warning("Can't find anything for video or pdf")
        return None
    
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# 交谈历史链
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# 处理用户输入问题
def handle_user_input(user_question):
    if user_question:
        if st.session_state.conversation is None:
            return
        
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        print('user_question is null')

def into_store(raw_text):
    # get the text chunks
    text_chunks = get_text_chunks(raw_text)

    # for c in text_chunks:
    #     print("\033[91m" + c + "\033[0m")
    
    # create vector store
    vectorstore = get_vectorstore(text_chunks)
    if (vectorstore is None):
        return

    # create conversation chain
    st.session_state.conversation = get_conversation_chain(vectorstore)

# 项目入口
def main():
    # load_dotenv()
    title = "AI Reader "
    icon = ":books:"

    # 设置html模版参数
    st.set_page_config(page_title=title,page_icon=icon)
    # 应用css
    st.write(css, unsafe_allow_html=True)

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # 设置标题
    st.header(title + icon)

    # 接收用户输入
    user_question = st.text_input("Ask a question about your files:")

    # 回答，相当于user_question的on_change处理
    if user_question:
        handle_user_input(user_question)
    
    # with语句适用于对资源进行访问的场合，它可以确保在对资源的操作过程中不管是否发生异常都会自动执行释放资源的操作，相当于try catch
    # 在sidebar里
    with st.sidebar:
        # 设置侧边栏标题
        st.subheader("Your documents")
        # 获取上传文件结果
        pdf_docs = st.file_uploader("Upload your Files(PDF，Video) here and click on 'Process'", accept_multiple_files=True)
        youtube_url = st.sidebar.text_area(label="What is the YouTube video URL?", max_chars=100 )
        website_url = st.sidebar.text_area(label="What is the Website URL?", max_chars=100 )
        api_key = st.text_input(label="Input openai api key", max_chars=100)

        # 按钮点击，把传入文件分析入库
        if st.button("Process"):
            api_key = api_key.strip()
            if api_key == "":
                st.warning("Please input your openai api key")
                return
            else:
                os.environ['OPENAI_API_KEY'] = api_key

            with st.spinner("Processing"):
                list_text=""
                # 如果有PDF
                if pdf_docs:
                    list_text += get_pdf_text(pdf_docs)
                # 如果有视频链接
                if youtube_url:
                    list_text += get_text_from_youtube_video_url(youtube_url)

                # 如果有网页链接
                if website_url:
                    list_text += get_text_from_website_url(website_url)

                # 合并分析
                into_store(list_text)
        
if __name__ == '__main__':
    main()

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
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_functions_agent
from langchain import hub
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from video import get_text_from_youtube_video_url
from website import get_text_from_website_url

message_history = ChatMessageHistory()

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
    # llm = ChatOpenAI()
    # # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    # memory = ConversationBufferMemory(
    #     memory_key='chat_history', return_messages=True)
    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=vectorstore.as_retriever(),
    #     memory=memory
    # )
    # return conversation_chain
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    retriever_tool = create_retriever_tool(
        vectorstore.as_retriever(),
        "my_search",
        "My custom search tool",
    )

    tools = [retriever_tool]

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/openai-functions-agent")
    # prompt.messages
    # prompt.messages[0].prompt.template = "You are a helpful assistant. When unsure, reply with \"Sorry, please contact our customer service, Tom.\""

    agent = create_openai_functions_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # message_history = ChatMessageHistory()

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # This is needed because in most real world scenarios, a session id is needed
        # It isn't really used here because we are using a simple in memory ChatMessageHistory
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_chat_history

# 处理用户输入问题
def handle_user_input(user_question):
    if user_question:
        if st.session_state.conversation is None:
            return
        
        response = st.session_state.conversation.invoke(
            {"input": user_question},
            config={"configurable": {"session_id": "<foo>"}},
        )
        print(response)

        for i, message in enumerate(response['chat_history']):
            print(message)
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
        
        st.write(bot_template.replace(
                    "{{MSG}}", response['output']), unsafe_allow_html=True)
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

        option = st.sidebar.radio(
            "Choose the model",
            ("Common", "News"),
            index=0
        )

        website_url = ''
        if option == "Common":
            website_url = st.sidebar.text_area(label="What is the Website URL?", max_chars=5000 )
        else:
            website_url = st.sidebar.text_area(label="What is the Website URL?", value="https://www.cnn.com,https://www.nytimes.com,https://www.bbc.com/news,https://www.theguardian.com,https://www.foxnews.com,https://www.nbcnews.com,https://www.dailymail.co.uk,https://www.washingtonpost.com,https://www.wsj.com,https://www.usatoday.com,https://www.espn.com,https://www.sports.yahoo.com,https://www.bleacherreport.com,https://www.cbssports.com,https://www.sportingnews.com,https://www.nbcsports.com,https://www.foxsports.com", max_chars=5000 )

        api_key = st.text_input(label="Input openai api key", max_chars=100)


        # 按钮点击，把传入文件分析入库
        if st.button("Process"):
            api_key = api_key.strip()
            if api_key == "":
                st.warning("Please input your openai api key")
                return
            elif api_key != "byron_demo":
                os.environ['OPENAI_API_KEY'] = api_key
            else:
                load_dotenv()
                os.environ['OPENAI_API_KEY'] = os.environ.get('INNER_OPENAI_API_KEY')

            with st.spinner("Processing"):
                list_text = ""
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

# 使用 Python 3.9 镜像作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制项目文件到工作目录
COPY . /app

# 安装项目依赖
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip
RUN pip install tiktoken
RUN pip install fake_useragent
RUN pip install streamlit --upgrade
RUN pip install -U langchain-community


# Create the Streamlit configuration directory
RUN mkdir -p ~/.streamlit

# Add configuration settings to hide "Deploy" button
RUN echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
port = 8501\n\
\n\
" > ~/.streamlit/config.toml

# Run the Streamlit app
#CMD ["streamlit", "run", "app.py", "--server.port", "8501"]

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

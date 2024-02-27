# 使用 Python 3.9 镜像作为基础镜像
FROM python:3.9

# 设置工作目录
WORKDIR /app

# 复制项目文件到工作目录
COPY . /app

# 安装项目依赖
RUN pip install -r requirements.txt
RUN pip install tiktoken
RUN pip install fake_useragent

# 暴露端口（如果你的应用需要）
EXPOSE 8501

# 启动 Streamlit 应用
CMD ["streamlit", "run", "app.py"]
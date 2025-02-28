# 使用 CUDA 11.8 + Ubuntu 22.04 基础镜像
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装 Miniconda3
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*
    
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# 配置 Conda 环境
ENV PATH="/miniconda3/bin:$PATH"
RUN conda create -n myenv python=3.10 && \
    echo "conda activate myenv" >> ~/.bashrc

# 安装依赖
COPY requirements.txt .
RUN /bin/bash -c "source activate myenv && \
    pip install --no-cache-dir -r requirements.txt"

# 复制项目代码
COPY ..\
WORKDIR ..\
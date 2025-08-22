FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL maintainer="UniboNLP"

# Zero interaction (default answers to all questions)
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /workdir
ENV APP_PATH=/workdir

# Install general-purpose dependencies
RUN apt-get update -y && \
    apt-get install -y curl \
                        git \
                        bash \
                        nano \
                        python3.11 \
                        python3-pip && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install wrapt --upgrade --ignore-installed
RUN pip install gdown


COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

RUN python3 -m nltk.downloader punkt punkt_tab
RUN pip install --upgrade --no-cache-dir vllm transformers
RUN pip install --upgrade --force-reinstall --no-deps --no-cache-dir unsloth unsloth_zoo

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog
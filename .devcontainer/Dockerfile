# https://kimi.moonshot.cn/share/curb2mtnfo2p8ffmnk4g

FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Install Miniconda
WORKDIR /app
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget --quiet -O - https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > /tmp/miniconda.sh
RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Install dependencies
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && rm /tmp/environment.yml
RUN conda init
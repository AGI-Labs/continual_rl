FROM ubuntu:20.04

# Setup git and git clone
RUN apt-get update       
RUN apt-get install -y git
RUN mkdir ~/Git && \
	cd ~/Git && \
	git clone https://github.com/AGI-Labs/continual_rl.git &&\
	cd continual_rl

# Setup conda, courtesy https://stackoverflow.com/questions/58269375/how-to-install-packages-with-miniconda-in-dockerfile
ENV PATH="/root/miniconda3/bin:$PATH"
ARG PATH="/root/miniconda3/bin:$PATH"
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda init bash 

RUN  conda create -y -n venv_cora python=3.8

RUN echo "source activate venv_cora" > ~/.bashrc

WORKDIR /root/Git/continual_rl
RUN pip install torch>=1.7.1 torchvision
RUN pip install -e .

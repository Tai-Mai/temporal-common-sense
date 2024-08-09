# "Base" of dev-Container
# FROM python:3.12-slim as python

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Install system applications for easier development
RUN apt update && apt install --no-install-recommends -y \
    sudo \
    git \
    neovim \
    zsh
#    ssh \
#    git-lfs \
#    libssl-dev \
#    curl \
#    wget \

# This has to be customized if your UNIX user's id != 1000
#RUN addgroup --gid 1002 developer
#RUN adduser --uid 1001 --ingroup developer --shell /bin/sh --disabled-login developer
# grant developer sudo privileges
#RUN sed -i 's|root ALL=(ALL:ALL) ALL|root ALL=NOPASSWD: ALL\ndeveloper ALL=NOPASSWD: ALL|g' /etc/sudoers

#RUN mkdir -p /home/developer/workspace && chown -R developer:developer /home/developer/workspace
#USER developer
#WORKDIR /home/developer/workspace
WORKDIR /app

# install oh-my-zsh
#RUN curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh | bash - && \
#  git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
#  git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \
#  sed -i 's|plugins=(git)|plugins=(git docker zsh-autosuggestions zsh-syntax-highlighting)|g' ~/.zshrc && \
#  sed -i 's|ZSH_THEME="robbyrussell"|ZSH_THEME="clean"|g' ~/.zshrc



# Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
#RUN pip install --upgrade transformers
#RUN git clone https://github.com/Dao-AILab/flash-attention.git
#WORKDIR flash-attention
#RUN python setup.py install
#WORKDIR /app

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN apt update && apt upgrade -y
RUN apt-get -y install software-properties-common
RUN add-apt-repository -y ppa:fish-shell/release-4
RUN apt update && apt upgrade -y
RUN apt-get -y install git fish tmux

RUN python -m pip install ffmpeg-python
RUN python -m pip install tqdm

RUN apt update && apt upgrade -y
RUN apt-get -y install curl
RUN apt update && apt upgrade -y
RUN apt-get install -y ffmpeg
RUN apt update && apt upgrade -y

RUN apt-get install -y libgl1-mesa-glx
RUN apt update && apt upgrade -y
RUN apt-get install -y libglib2.0-0
RUN apt update && apt upgrade -y

RUN python -m pip install opencv-python==4.10.0.84
RUN python -m pip install matplotlib
RUN python -m pip install decord

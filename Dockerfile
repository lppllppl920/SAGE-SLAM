# Import the docker base image
FROM nvcr.io/nvidia/pytorch:21.04-py3

# Define arguments, username, user id, group id, and user password, whose values will be provided later in the docker build command
ARG UNAME
ARG UID
ARG GID
ARG PW

# Install system dependencies
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y sudo python3-pip git unzip wget cmake cmake-gui gdb htop ffmpeg
RUN DEBIAN_FRONTEND=noninteractive apt install -y libsm6 libxext6 libxrender-dev libtbb-dev
RUN DEBIAN_FRONTEND=noninteractive apt install -y libboost-all-dev libglew-dev libgoogle-glog-dev libjsoncpp-dev libopenni2-dev libavcodec-dev libavutil-dev libavformat-dev
RUN DEBIAN_FRONTEND=noninteractive apt install -y libswscale-dev libavdevice-dev libjpeg-dev libpng-dev libtiff5-dev libopenexr-dev libopenblas-base libopenblas-dev libglu1-mesa-dev
RUN DEBIAN_FRONTEND=noninteractive apt install -y freeglut3-dev mesa-common-dev libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libncurses-dev
RUN DEBIAN_FRONTEND=noninteractive apt install -y libswscale-dev libhdf5-dev libgflags-dev libboost-all-dev libprotobuf-dev protobuf-compiler
RUN python3 -m pip install --upgrade --force pip

# libopencv-dev
# Set group, user and password and grant the user 'sudo' priviledge
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
RUN adduser $UNAME sudo
RUN echo "$UNAME:$PW" | chpasswd

RUN pip3 install tqdm==4.53.0 && \
  pip3 install torchgeometry==0.1.2 && \
  pip3 install graphviz==0.19.1 && \
  pip3 install umap-learn==0.5.2

# Install generic python packages
RUN pip install autopep8

# This is used to fix some issues related to autopep8
RUN chmod o+rw /opt/conda

# Run following commands with user priviledge instead of root
USER $UNAME

# Set the path variable to include the user directory
ENV PATH="/home/$UNAME/.local/bin:${PATH}"

# The following two commands for python-related environment variables are optional
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Change the default working directory of the final docker image to the home directory within the image
WORKDIR /home/$UNAME

LABEL developer="Xingtong Liu <xliu89jhu@gmail.com>"

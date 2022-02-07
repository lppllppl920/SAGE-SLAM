# Import the docker base image of your choice
FROM nvcr.io/nvidia/pytorch:21.04-py3

# Define arguments, username, user id, group id, and user password, whose values will be provided later in the docker build command
ARG UNAME
ARG UID
ARG GID
ARG PW

# Install system dependencies of your choice, here we install sudo and python-related dependencies
RUN apt update && \
  apt install -y sudo && \
  apt install -y python3-pip git libsm6 libxext6 libxrender-dev && \
  python3 -m pip install --upgrade --force pip

# Set group, user and password and grant the user 'sudo' priviledge
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
RUN adduser $UNAME sudo
RUN echo "$UNAME:$PW" | chpasswd

# # Copy private repository from the intermediate docker image to /libraries folder in the final docker image
# RUN chown $UNAME:$UNAME /libraries
# # One example for copying newton library: COPY --from=intermediate --chown=$UNAME:$UNAME /app/newton /libraries/newton
# COPY --from=intermediate --chown=$UNAME:$UNAME /app/<library name> /libraries/<library name>

# Run following commands with user priviledge instead of root
# USER $UNAME

RUN pip3 install tqdm==4.53.0 && \
  pip3 install opencv-python==4.2.0.34 && \
  pip3 install torchgeometry==0.1.2 && \
  pip3 install graphviz==0.19.1 && \
  pip3 install umap-learn==0.5.2

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

# docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg UNAME=$(whoami) --build-arg PW=password -f Dockerfile -t sage-slam .
# docker run -it --gpus=all --ipc=host --mount type=bind,source="/home/xingtong/sage-slam",target="/home/xingtong/" --mount type=bind,source="/tmp/",target="/tmp/" --name sage-slam sage-slam

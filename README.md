# SAGE-SLAM


1. Clone this repository.
2. Download an example dataset from [this link](https://drive.google.com/drive/folders/1dYC_-w4HsrmA5ecQ3a_2R0FnJX0Bmc-o?usp=sharing).
3. Create a `data` folder inside the cloned repository and put the downloaded folder `bag_1` inside the `data` folder.
4. Build a Docker image and start a Docker container created from the built Docker image.
    ```
    docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg UNAME=$(whoami) --build-arg PW=password -f Dockerfile -t sage-slam .
    ```

    ```
    docker run -it --gpus=all --ipc=host --mount type=bind,source="<root of the cloned repository>",target="<home directory within the docker container>" --mount type=bind,source="/tmp/",target="/tmp/" --name sage-slam sage-slam
    ```
5. Now you should be at the home directory of the Docker container. Enter the `representation` folder and run the following command for training
    ```
    python3 "./training.py" --config_path "./configs/training.json"
    ```
    Note that a set of pre-trained network models are provided inside the `pretrained` folder and they are loaded with the settings specified in the    `training.json`. Set `net_load_weights` within `training.json` to `false` if you want to train the networks from scratch.



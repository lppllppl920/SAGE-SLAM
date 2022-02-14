# SAGE-SLAM


## Instructions

1. Clone this repository with 
    ```
    git clone git@github.com:lppllppl920/SAGE-SLAM.git
    ```
2. Download an example dataset from [this link](https://drive.google.com/drive/folders/1dYC_-w4HsrmA5ecQ3a_2R0FnJX0Bmc-o?usp=sharing).
3. Create a `data` folder inside the cloned repository and put the downloaded folder `bag_1` inside the `data` folder.
4. After the steps above, the folder structure of the cloned repository will be shown as below with the command `tree -d -L 2 <path of the cloned repository>`
    ```
    ├── data
    │   └── bag_1
    ├── pretrained
    ├── representation
    │   ├── configs
    │   ├── datasets
    │   ├── losses
    │   ├── models
    │   ├── scripts
    │   └── utils
    └── system
        ├── configs
        ├── sources
        └── thirdparty
    ```

5. Build a Docker image and start a Docker container created from the built Docker image. Note that the `PW` in the first command can be specified as any string as the password to access the `sudo` priviledge inside the Docker container.
    ```
    cd <path of the cloned repository> && \
    docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg UNAME=$(whoami) --build-arg PW=<password of your choice> -f Dockerfile -t sage-slam . && \
    docker run -it --gpus=all --ipc=host --mount type=bind,source=<path of the cloned repository>,target=$HOME --mount type=bind,source=/tmp,target=/tmp --name sage-slam sage-slam
    ```
    
6. Now you should be at the home directory of the Docker container. To start the representation learning process, run the following command:
    ```
    cd $HOME && \
    /opt/conda/bin/python $HOME/representation/training.py --config_path "$HOME/representation/configs/training.json"
    ```
    Note that a set of pre-trained network models are provided inside the `pretrained` folder of the repository. With the given setting specified in the `$HOME/representation/configs/training.json`, these pre-trained models are loaded. Set `net_load_weights` inside the `training.json` to `false` if you want to train the networks from scratch.

7. To generate Pytorch JIT ScriptModule's that will be used in the SAGE-SLAM system, change `net_depth_model_path`, `net_feat_model_path`, `net_ba_model_path`, and `net_disc_model_path` inside `$HOME/representation/configs/export.json` to the corresponding model paths and run the following command:
    ```
    cd $HOME && \
    /opt/conda/bin/python $HOME/representation/training.py --config_path "$HOME/representation/configs/export.json" 
    ```

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

5. Install the Docker Engine with the instructions [here](https://docs.docker.com/engine/install/), build a Docker image, and start a Docker container created from the built Docker image. Note that the `PW` in the `docker build` command can be specified as any string as the password to access the `sudo` priviledge inside the Docker container. Note that the step 6, 7, and 8 below are optional if you only want to test run the SAGE-SLAM system, because we have pre-generated all required data. 
    ```
    cd <path of the cloned repository> && \
    docker build \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    --build-arg UNAME=$(whoami) \
    --build-arg PW=<password of your choice> \
    -f Dockerfile \
    -t sage-slam \
    . && \
    docker run \
    -it \
    --privileged \
    --env DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $HOME/.Xauthority:$HOME/.Xauthority:rw \
    --gpus=all \
    --ipc=host \
    --net=host \
    --mount type=bind,source=<path of the cloned repository>,target=$HOME \
    --mount type=bind,source=/tmp,target=/tmp \
    --name sage-slam \
    sage-slam
    ```
    Note that some of the options in the `docker run` command are to enable X11 display inside the Docker container. Run `sudo apt install -y firefox` and `firefox` within the container to install the firefox browser and open it up to test if the X11 display is working normally. Recent versions MacOS seem to have problems supporting the X11 display used by the third-party library `Pangolin` of this repository. In this case, the GUI can be disabled when the SLAM system is ran, which is introduced later.
    
6. Now the current working directory should be the home directory of the Docker container. To start the representation learning process, run the following command:
    ```
    cd $HOME && \
    /opt/conda/bin/python $HOME/representation/training.py \
    --config_path "$HOME/representation/configs/training.json"
    ```
    Note that a set of pre-trained network models are provided inside `$HOME/pretrained` folder. With the given setting specified in the `$HOME/representation/configs/training.json`, these pre-trained models are loaded. Set `net_load_weights` inside the `training.json` to `false` if you want to train the networks from scratch.
    
7. To visualize the tensorboard outputs during the training process, open a new terminal console that is outside of the Docker container, and run the following command:
    ```
    tensorboard --logdir="/tmp/SAGE-SLAM_<time of the experiment>" \
    --host=127.0.0.1 \
    --port=6006
    ```
    Then open a compatible browser (such as Google Chrome) and type in `http://localhost:6006/` to open the tensorboard dashboard. Note that the value of the option `logdir` should be the path of the experiment of which you want to inspect the results.

8. Inside the Docker container, to generate Pytorch JIT ScriptModule's that will be used in the SAGE-SLAM system, change `net_depth_model_path`, `net_feat_model_path`, `net_ba_model_path`, and `net_disc_model_path` inside `$HOME/representation/configs/export.json` to the corresponding model paths and run the following command:
    ```
    cd $HOME && \
    /opt/conda/bin/python $HOME/representation/training.py \
    --config_path "$HOME/representation/configs/export.json" 
    ```
    
9. To build the SAGE-SLAM system implemented in C++, run the following command:
    ```
    SLAM_BUILD_TYPE=Release && \
    $HOME/system/thirdparty/makedeps_with_argument.sh $SLAM_BUILD_TYPE && \
    mkdir -p $HOME/build/$SLAM_BUILD_TYPE && \
    cd $HOME/build/$SLAM_BUILD_TYPE && \
    cmake -DCMAKE_BUILD_TYPE=$SLAM_BUILD_TYPE $HOME/system/ && \
    make -j4 && \
    cd $HOME
    ```
    Note the `SLAM_BUILD_TYPE` can be changed to `Debug` to enable debugging if you want to further develop the SLAM system. With this command executed, the folder structure within the Docker container should look like below with the command `tree -d -L 3 $HOME`:
    ```
    ├── build
    │   └── Release
    │       ├── bin
    │       ├── CMakeFiles
    │       ├── sources
    │       └── thirdparty
    ├── data
    │   └── bag_1
    │       ├── _start_002603_end_002984_stride_1000_segment_00
    │       ├── _start_003213_end_003527_stride_1000_segment_00
    │       └── _start_004259_end_004629_stride_1000_segment_00
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
        │   ├── common
        │   ├── core
        │   ├── cuda
        │   ├── demo
        │   ├── drivers
        │   ├── gui
        │   └── tools
        └── thirdparty
            ├── build_Release
            ├── camera_drivers
            ├── DBoW2
            ├── eigen
            ├── gtsam
            ├── install_Release
            ├── opengv
            ├── Pangolin
            ├── Sophus
            ├── TEASER-plusplus
            └── vision_core
    ```
    
10. Run the SAGE-SLAM system with the following command:
    ```
    SLAM_BUILD_TYPE=Debug && \
    cd $HOME && \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/system/thirdparty/install_$SLAM_BUILD_TYPE/lib \
    $HOME/build/$SLAM_BUILD_TYPE/bin/df_demo \
    --flagfile $HOME/system/configs/slam_run.flags \
    --enable_gui=false
    ```
    Note that if the X11 display is working normally, the option `enable_gui` can be set to `true` to see if the GUI of the SLAM system can be brought up. The visualization inside the Docker container has not been fully tested for this SLAM system and please let us know if there is any issue with it.


## Contact Us

[Xingtong Liu](https://www.linkedin.com/in/xingtong-liu-ph-d-b43b27131/) ([xliu89jhu@gmail.com](mailto:xliu89jhu@gmail.com))

[Mathias Unberath](https://engineering.jhu.edu/faculty/mathias-unberath/) ([unberath@jhu.edu](mailto:unberath@jhu.edu))


# Are Labels Always Necessary for Classifier Accuracy Evaluation? : USPS test

## Prerequisites
- docker : verson 20.10.8

## Dataset
- MSCOCO, MNIST, USPS
- I use usps.h5 in [this cite](https://www.kaggle.com/bistaumanga/usps-dataset)
- ex : tree of dataset directory
    ```bash
    dataset
    |-- MSCOCO/train2014 
    |-- mnist
    |-- usps
    `-- |-- usps.h5

    ```

## Getting started
0. build docker image and run container
    ```bash
    bash sh/docker_run.sh PATH/TO/MSCOCO PATH/TO/MNIST PATH/TO/USPS IMAGE_NAME CONTAINER_NAME PATH/TO/DATASET/ROOT PATH/TO/WORK/DIR
    ```
    - ex
        ```bash
        bash sh/docker_run.sh /dataset/MSCOCO/train2014 /dataset/mnist /dataset/usps autoeval_usps autoeval_usps_1 /dataset /home/Meta-set-usps
        ```
        - `PATH/TO/MSCOCO` : /dataset/MSCOCO/train2014
        - `PATH/TO/MNIST` : /dataset/mnist
        - `PATH/TO/USPS` : /dataset/usps
        - `IMAGE_NAME` : autoeval_usps
        - `CONTAINER_NAME` : autoeval_usps_1
        - `PATH/TO/DATASET/ROOT` : /dataset
        - `PATH/TO/WORK/DIR` : /home/Meta-set-usps

1. run AutoEval in docker container (0,1,2,3,4,6 in original repository)
    ```bash
    bash sh/run.sh
    ```
2. check result
    ```bash
    tail results/TIME_STR.txt
    ```
    - ex
        ```bash
        tail results/2021-10-24-16-13.txt
        ```
        - `TIME_STR` : 2021-10-24-16-13
        - result
            ```bash
            epoch: 209
            Test USPS loss: 72.0426 

            target_acc: [tensor([13.7841]), tensor([13.1540])]

            pred_acc: [tensor([86.0173]), tensor([86.0059])]
            ```
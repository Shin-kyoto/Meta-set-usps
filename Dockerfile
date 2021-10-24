FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
ENV DEBIAN_FRONTEND=noninteractive

ENV ROOT /Meta-set
WORKDIR $ROOT

RUN pip install cython scipy==1.1.0 scikit-learn matplotlib tensorboard pandas seaborn h5py mlflow
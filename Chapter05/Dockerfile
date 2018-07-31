#base image provides CUDA support on Ubuntu 16.04
FROM nvidia/cuda:9.0-cudnn7-devel

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH
ENV NB_USER keras
ENV NB_UID 1000

#package updates to support conda
RUN apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz

#add on conda python and make sure it is in the path
RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet --output-document=miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /miniconda.sh -f -b -p $CONDA_DIR && \
    rm miniconda.sh

#setting up a user to run conda
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown keras $CONDA_DIR -R && \
    mkdir -p /src && \
    chown keras /src

#all the code
COPY . /src

#packages needed by our server
RUN pip install -r /src/requirements.txt

#train the machine learning model
RUN cd /src && python train_mnist.py

#serve up a jupyter notebook 
USER keras
WORKDIR /src
EXPOSE 5000
CMD cd /src && FLASK_ENV=development python server.py
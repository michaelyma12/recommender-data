FROM openjdk:8

RUN update-ca-certificates -f \
  && apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y \
    software-properties-common \
    wget \
    git \
    libatlas3-base \
    libopenblas-base \
    libatlas-base-dev \
    build-essential \
  && apt-get clean

# install spark
ENV SPARK_VERSION=3.0
ENV HADOOP_VERSION=2.7.4

RUN wget --no-verbose https://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop2.7.tgz
RUN tar -xzf /spark-3.0.0-bin-hadoop2.7.tgz && \
 mv spark-3.0.0-bin-hadoop2.7 spark && \
 echo "export PATH=$PATH:spark/bin" >> ~/.bashrc
RUN wget https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-latest-hadoop2.jar -P /spark/jars/

ENV SPARK_HOME /spark
ENV SPARK_MAJOR_VERSION 3
ENV PYTHONPATH=$SPARK_HOME/python/lib/py4j-0.10.7-src.zip:$SPARK_HOME/python/:$PYTHONPATH

# install conda
ENV CONDA_DIR /opt/miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod a+x miniconda.sh && \
    ./miniconda.sh -b -p $CONDA_DIR && \
    rm ./miniconda.sh
ENV PATH="$CONDA_DIR/bin/":$PATH

RUN pip install --upgrade pip \
  && pip install pylint coverage pytest black --quiet
ENV PATH=$PATH:$SPARK_HOME/bin/

# set up conda environment
ADD environment.yml /tmp/environment.yml
RUN mkdir -p recommender-data
ADD .gcp /recommender-data/.gcp
WORKDIR recommender-data

# setup necessary directory structure
COPY ./pipeline ./pipeline
COPY ./deploy ./deploy
RUN mkdir -p models/candidate_generation


# run commands from recommendations env
RUN conda env create -f /tmp/environment.yml
RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" >> ~/.bashrc
SHELL ["conda", "run", "-n", "recommender-data", "/bin/bash", "-c"]

# export pyspark ENV VARS
RUN export PATH=/opt/miniconda/bin:$PATH
RUN export PYSPARK_PYTHON=/opt/miniconda/envs/recommender-data/bin/python3
RUN export PYSPARK_DRIVER_PYTHON=/opt/miniconda/envs/recommender-data/bin/python3

# run recommender
#CMD /opt/miniconda/envs/recommender-data/bin/python3.7 deploy/deploy_cg_data.py

FROM ubuntu:18.04
ENV WORKDIR=/svm_liblinear

# To beautifully print utf-8 characters
ENV PYTHONIOENCODING utf-8
ENV LANG="C.UTF-8"
ENV JOBLIB_TEMP_FOLDER=/tmp
ENV TMPDIR=/tmp
ENV DEBIAN_FRONTEND=teletype
ENV TZ=America/Sao_Paulo

COPY . /svm_liblinear/docker
WORKDIR $WORKDIR

#Install python and other modules
RUN bash docker/install_base.sh

#Installing dependences
RUN bash docker/install_requirements.sh

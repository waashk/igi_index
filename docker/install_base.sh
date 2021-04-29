#Install python and other modules
echo "Installing python..."
apt-get update --fix-missing -y && apt-get install -y --no-install-recommends apt-utils module-init-tools software-properties-common build-essential && \
                          add-apt-repository ppa:deadsnakes/ppa && \
                          apt-get update && \
                          apt-get install -y python3.6 python3.6-dev python3-pip && \
                          apt-get install -y --no-install-recommends module-init-tools wget nano curl git gnuplot

echo "Linking python..."
ln -sfn /usr/bin/python3.6 /usr/bin/python3 && \
    ln -sfn /usr/bin/python3 /usr/bin/python && \
    ln -sfn /usr/bin/pip3 /usr/bin/pip

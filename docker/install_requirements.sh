#Install python and other modules
cd $WORKDIR
echo "Upgrading setuptools pip wheel"
#pip-21.0.1 setuptools-54.2.0 wheel-0.36.2
python -m pip install --upgrade setuptools pip wheel
echo "Installing requeriments"
python -m pip install -r docker/requirements.txt 

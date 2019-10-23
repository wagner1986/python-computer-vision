FROM ufoym/deepo:all-py36-jupyter-cpu
ADD . /source
WORKDIR /source
RUN pip install -r requirements.txt
CMD jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --NotebookApp.allow_origin='*' --port=8880
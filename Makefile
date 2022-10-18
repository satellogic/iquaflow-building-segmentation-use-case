PROJ_NAME=iqf-tf-test# set name
DATA_VOLUME = $(pwd)# set path to folder to mount in docker
CONTAINER_NAME="${PROJ_NAME}-${USER}"


GEN_PORT=9089
NB_PORT=8188
MLF_PORT=5045

help:
	@echo "build -- builds the docker image"
	@echo "container -- runs a container in the background"
	@echo "nb -- launches a notebook server"
	@echo "nbstop -- Stops the notebook server"
	@echo "mlf -- launches an mlflow server"
	@echo "execsh -- raises an interactive shell docker from a running container"
	@echo "test -- run tests"
	@echo "stop -- Stop the docker container and everything that is running within it"
	@echo "downloads course materials"

build:
	docker build -t ${PROJ_NAME} .
	./download.sh
	

container:
	docker run --rm -itd --name $(CONTAINER_NAME) --gpus all --privileged \
	-p $(GEN_PORT):$(GEN_PORT) \
	-p $(NB_PORT):$(NB_PORT) \
	-p $(MLF_PORT):$(MLF_PORT) \
	-v $(DATA_VOLUME):$(DATA_VOLUME) \
	-w $(shell pwd) \
	${PROJ_NAME} /bin/bash

nb:
	docker exec -d --privileged $(CONTAINER_NAME) \
	jupyter lab \
	--NotebookApp.token='iqf' \
	--no-browser \
	--ip=0.0.0.0 \
	--allow-root \
	--port=$(NB_PORT)

nbstop:
	docker exec -d --privileged $(CONTAINER_NAME) \
	jupyter lab stop ${NB_PORT}

mlf:
	docker exec -d $(CONTAINER_NAME) mlflow ui --host 0.0.0.0:$(MLF_PORT)

execsh:
	docker exec -it ${CONTAINER_NAME} /bin/bash

test:
	docker exec -it -w $(shell pwd) ${CONTAINER_NAME} pip3 install pytest flake8
	docker exec -it -w $(shell pwd) ${CONTAINER_NAME} pytest tests/

stop:
	docker stop $(CONTAINER_NAME)

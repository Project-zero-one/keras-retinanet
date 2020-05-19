docker run \
	--gpus all \
	-itd \
	--rm \
	-p $2:$2 \
	--name $1 \
	--mount type=bind,source="$PWD",target=/app \
	--mount type=bind,source=/home,target=/home \
	--mount type=bind,source=/media,target=/media \
	--mount type=bind,source=/data,target=/data \
	tensorflow/tensorflow:1.14.0-gpu-py3 \
	bash

docker exec -it $1 bash
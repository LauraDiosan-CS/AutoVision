FROM python:3.13-bookworm

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc
ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt-get update && apt-get install --no-install-recommends -y python3-pip python3-opencv python3-matplotlib
RUN pip install PyQt6 ultralytics polars maturin --break-system-packages

COPY . vision/

WORKDIR /vision/rs_ipc
RUN maturin build -r && pip install target/wheels/rs_ipc*.whl --break-system-packages

WORKDIR /vision
###build until here with docker build . -t "carvision"

### run container and attach with
# docker run -it --name carvision --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix" <Image_id> /bin/bash

### Wayland possible solution?  # NOT TESTED
#docker run --name <container_name> --env="WAYLAND_DISPLAY=$WAYLAND_DISPLAY" --env="XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" \--volume="$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY" <IMAGE ID> bash

##execute in bash ```python3 main.py

FROM python:3.14

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc
ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt-get update && apt-get install -y \
    # --- GUI Dependencies Start --- \
    libx11-dev \
    libx11-xcb-dev \
    libxcursor-dev \
    libxrandr-dev \
    libxi-dev \
    libxcb-shape0-dev \
    libxcb-xfixes0-dev \
    libxkbcommon-x11-dev \
    libgl1 \
    # --- GUI Dependencies End --- \
    npm \
    sccache \
    && rm -rf /var/lib/apt/lists/*

RUN pip install ultralytics maturin opencv-python

ENV RUSTC_WRAPPER=sccache SCCACHE_DIR=/sccache

COPY . vision/

WORKDIR /vision/rs_ipc
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    maturin build -r && pip install target/wheels/rs_ipc*.whl


WORKDIR /vision
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo build -r --manifest-path rs_ui/Cargo.toml

ENTRYPOINT ["rs_ui/target/release/rs_ui"]

# To run the container run: xhost +local:docke

# docker build . -t vision
# docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --ulimit nofile=65536:65536 --ipc=host -it --rm vision
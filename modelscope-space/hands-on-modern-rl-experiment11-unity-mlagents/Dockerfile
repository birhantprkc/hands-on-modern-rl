FROM modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.1.0-py310-torch2.3.1-tf2.16.1-1.20.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    UNITY_DISPLAY=:99 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    MESA_GL_VERSION_OVERRIDE=4.5 \
    QT_X11_NO_MITSHM=1 \
    SDL_AUDIODRIVER=dummy \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        aria2 \
        ca-certificates \
        curl \
        ffmpeg \
        libasound2 \
        libgl1 \
        libgl1-mesa-dri \
        libglib2.0-0 \
        libglu1-mesa \
        libnss3 \
        libvulkan1 \
        libx11-6 \
        libx11-xcb1 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxi6 \
        libxinerama1 \
        libxrandr2 \
        libxrender1 \
        libxtst6 \
        mesa-utils \
        unzip \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/studio/app

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip install --no-cache-dir --ignore-requires-python \
        mlagents==1.1.0 \
        mlagents-envs==1.1.0

COPY . /home/studio/app

EXPOSE 7860

ENTRYPOINT ["python", "-u", "app.py"]

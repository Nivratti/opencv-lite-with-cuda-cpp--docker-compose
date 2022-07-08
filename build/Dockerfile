# https://learnopencv.com/opencv-dnn-with-gpu-support/
## for optimization: https://github.com/JulianAssmann/opencv-cuda-docker/blob/master/ubuntu-20.04/opencv-4.5/cuda-11.1/Dockerfile

FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ARG OPENCV_VERSION=4.6.0

# Update the system so that the public apt repositry locations are visible
RUN apt update -y && apt upgrade -y

RUN apt-get update && apt-get upgrade -y && \
    # Install build tools, build dependencies and python
    apt-get install -y \
        build-essential \
        cmake \
        unzip \
        pkg-config \
        git \
        curl \
        ca-certificates \
        sudo \
        git \
        bzip2 \
        libx11-6 \
        build-essential \ 
        pkg-config \ 
        ## Python
        python3-pip \
        python3-dev \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# | Install open cv version 4.5.5|
# ---------------------------

## If static linking then remove libpng system package otherwise opencv will not install it use systems shared library and it will cause issue in cmake
## build for library 'opencv_imgcodecs' --- more info -- libpng installed by apt install libgtk-3-dev
## ; exit 0 -- Ignore error is package not exists
RUN apt remove libpng-dev; exit 0

# BUILD_SHARED_LIBS=OFF -- for building static files
# WITH_OPENCL=OFF -- turn off unused package from build
# And turn off all Video I/O utils -- from -WITH_V4L to -DWITH_1394
RUN cd && mkdir opencv_build && cd opencv_build \
    && git clone --progress --verbose https://github.com/opencv/opencv.git \
    && git clone --progress --verbose https://github.com/opencv/opencv_contrib.git \
    && cd ~/opencv_build/opencv && git checkout --progress tags/$OPENCV_VERSION && \
    cd ~/opencv_build/opencv_contrib && git checkout --progress tags/$OPENCV_VERSION && \
    # Create build folder and switch to it
    mkdir ~/opencv_build/opencv/build && cd ~/opencv_build/opencv/build \
    # Cmake configure
	&& cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D BUILD_SHARED_LIBS=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D INSTALL_C_EXAMPLES=OFF \
        -D BUILD_ZLIB=ON \
        -D WITH_OPENCL=OFF \
        -D WITH_CUDA=ON \
        -D WITH_CUDNN=ON \
        -D WITH_CUBLAS=ON \
        -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.0 \
        -D OpenCL_LIBRARY=/usr/local/cuda-11.0/lib64/libOpenCL.so \
        -D OpenCL_INCLUDE_DIR=/usr/local/cuda-11.0/include/ \
        -D PYTHON_EXECUTABLE=/usr/bin/python3 \
        -D WITH_V4L=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF \
        -D WITH_MSMF=OFF -DWITH_DSHOW=OFF -DWITH_AVFOUNDATION=OFF -DWITH_1394=OFF \
        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules .. \
    # Make
    && make -j$(nproc) \
    # Install to /usr/local/lib
    && make install \
    && ldconfig \
    # Remove OpenCV sources and build folder
    && rm -rf ~/opencv_build

# 
# HPC Base image
# 
# Contents:
#   GCC 7
#   FFTW version 3.3.8
#   HDF5 version 1.10.6
#   Mellanox OFED version 5.0-2.1.8.0
#   MVAPICH version 2.3.3
#   NVIDIA HPC SDK version 20.7
#   Python 2 and 3 (upstream)
#   Cmake 3.19.0-rc3
#   Boost (latest)
# 

FROM centos:7 AS devel

# Python
RUN yum install -y \
        python2 \
        python3 && \
    rm -rf /var/cache/yum/*

# GNU compiler
RUN yum install -y centos-release-scl && \
    yum install -y \
        devtoolset-7-gcc \
        devtoolset-7-gcc-c++ \
        devtoolset-7-gcc-gfortran && \
    rm -rf /var/cache/yum/*
RUN update-alternatives --install /usr/bin/g++ g++ /opt/rh/devtoolset-7/root/usr/bin/g++ 30 && \
    update-alternatives --install /usr/bin/gcc gcc /opt/rh/devtoolset-7/root/usr/bin/gcc 30 && \
    update-alternatives --install /usr/bin/gcov gcov /opt/rh/devtoolset-7/root/usr/bin/gcov 30 && \
    update-alternatives --install /usr/bin/gfortran gfortran /opt/rh/devtoolset-7/root/usr/bin/gfortran 30

# NVIDIA HPC SDK version 20.7
RUN yum install -y \
        libatomic \
        numactl-libs \
        openssh-clients \
        wget \
        which && \
    rm -rf /var/cache/yum/*
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://developer.download.nvidia.com/hpc-sdk/20.7/nvhpc_2020_207_Linux_x86_64_cuda_multi.tar.gz && \
    mkdir -p /var/tmp && tar -x -f /var/tmp/nvhpc_2020_207_Linux_x86_64_cuda_multi.tar.gz -C /var/tmp -z && \
    cd /var/tmp/nvhpc_2020_207_Linux_x86_64_cuda_multi && NVHPC_ACCEPT_EULA=accept NVHPC_INSTALL_DIR=/opt/nvidia/hpc_sdk NVHPC_SILENT=true ./install && \
    rm -rf /var/tmp/nvhpc_2020_207_Linux_x86_64_cuda_multi /var/tmp/nvhpc_2020_207_Linux_x86_64_cuda_multi.tar.gz
ENV LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/20.7/comm_libs/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/20.7/comm_libs/nccl/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/20.7/math_libs/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/20.7/compilers/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/20.7/cuda/lib64:$LD_LIBRARY_PATH \
    MANPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/20.7/compilers/man:$MANPATH \
    PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/20.7/comm_libs/nvshmem/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/20.7/comm_libs/nccl/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/20.7/profilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/20.7/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/20.7/cuda/bin:$PATH

# Mellanox OFED version 5.0-2.1.8.0
RUN yum install -y \
        ca-certificates \
        gnupg \
        wget && \
    rm -rf /var/cache/yum/*
RUN rpm --import https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox && \
    yum install -y yum-utils && \
    yum-config-manager --add-repo https://linux.mellanox.com/public/repo/mlnx_ofed/5.0-2.1.8.0/rhel7.2/mellanox_mlnx_ofed.repo && \
    yum install -y \
        libibumad \
        libibverbs \
        libibverbs-utils \
        librdmacm \
        rdma-core \
        rdma-core-devel && \
    rm -rf /var/cache/yum/*


# MVAPICH2 version 2.3.3
RUN yum install -y \
        byacc \
        file \
        flex \
        make \
        openssh-clients \
        wget && \
    rm -rf /var/cache/yum/*
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp http://mvapich.cse.ohio-state.edu/download/mvapich/mv2/mvapich2-2.3.3.tar.gz && \
    mkdir -p /var/tmp && tar -x -f /var/tmp/mvapich2-2.3.3.tar.gz -C /var/tmp -z && \
    cd /var/tmp/mvapich2-2.3.3 &&  CC=nvc CFLAGS='-fpic -DPIC' CXX=nvc++ F77=nvfortran FC=nvfortran FCFLAGS='-fpic -DPIC' FFLAGS='-fpic -DPIC' ./configure --prefix=/usr/local/mvapich2 --disable-cuda --disable-mcast ac_cv_c_compiler_gnu=no && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/mvapich2-2.3.3 /var/tmp/mvapich2-2.3.3.tar.gz
ENV LD_LIBRARY_PATH=/usr/local/mvapich2/lib:$LD_LIBRARY_PATH \
    PATH=/usr/local/mvapich2/bin:$PATH

# Git
RUN yum install -y git git-lfs && \
    rm -rf /var/cache/yum/*

RUN git clone https://github.com/LeviBarnes/cufft_benchmark benchmark && chown -R 9900:9900 benchmark

# FFTW version 3.3.8
RUN yum install -y \
        file \
        make \
        wget && \
    rm -rf /var/cache/yum/*
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp ftp://ftp.fftw.org/pub/fftw/fftw-3.3.8.tar.gz && \
    mkdir -p /var/tmp && tar -x -f /var/tmp/fftw-3.3.8.tar.gz -C /var/tmp -z && \
    cd /var/tmp/fftw-3.3.8 &&  CC=nvc CXX=nvc++ F77=nvfortran F90=nvfortran FC=nvfortran ./configure --prefix=/usr/local/fftw --enable-mpi --enable-openmp --enable-shared --enable-sse2 --enable-threads && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/fftw-3.3.8 /var/tmp/fftw-3.3.8.tar.gz
ENV LD_LIBRARY_PATH=/usr/local/fftw/lib:$LD_LIBRARY_PATH

ENV PATH=/usr/local/bin:$PATH

ENV INCLUDES="-I/opt/nvidia/hpc_sdk/Linux_x86_64/20.7/math_libs/include/ -I/opt/nvidia/hpc_sdk/Linux_x86_64/20.7/cuda/11.0/include/" \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=11.0 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411" \
    NVIDIA_VISIBLE_DEVICES=all

WORKDIR /benchmark

# syntax=docker/dockerfile:1.4
# Copyright (c) 2020 The Regents of the University of California
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
FROM ghcr.io/psal-postech/torchsim_base:latest

# Pass Access Token securely
ARG GEM5_ASSET_ID
ARG LLVM_ASSET_ID
ARG TORCHSIM_SHA
ENV PATH $PATH:/root/.local/bin
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/opt/conda/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

# Download GEM5 for torchsim
RUN --mount=type=secret,id=GIT_ACCESS_TOKEN \
    GIT_ACCESS_TOKEN=$(cat /run/secrets/GIT_ACCESS_TOKEN) && \
    curl -L -H "Accept: application/octet-stream" -H "Authorization: Bearer ${GIT_ACCESS_TOKEN}" https://api.github.com/repos/PSAL-POSTECH/gem5/releases/assets/${GEM5_ASSET_ID} -o /tmp/gem5-release.tar.gz && \
    mkdir -p /gem5 && \
    tar -xzf /tmp/gem5-release.tar.gz -C /gem5 && \
    rm /tmp/gem5-release.tar.gz
ENV GEM5_PATH /gem5/release/gem5.opt

# Download LLVM RISC-V for torchsim
RUN --mount=type=secret,id=GIT_ACCESS_TOKEN \
    GIT_ACCESS_TOKEN=$(cat /run/secrets/GIT_ACCESS_TOKEN) && \
    curl -L -H "Accept: application/octet-stream" -H "Authorization: Bearer ${GIT_ACCESS_TOKEN}"  https://api.github.com/repos/PSAL-POSTECH/llvm-project/releases/assets/${LLVM_ASSET_ID} -o /tmp/riscv-llvm-release.tar.gz && \
    tar -xzf /tmp/riscv-llvm-release.tar.gz -C / && \
    rm /tmp/riscv-llvm-release.tar.gz

# Store RISC-V LLVM for TorchSim
ENV TORCHSIM_LLVM_PATH /riscv-llvm/bin
ENV TORCHSIM_LLVM_INCLUDE_PATH /riscv-llvm/include
ENV TORCHSIM_DIR /workspace/PyTorchSim
ENV LLVM_DIR /riscv-llvm

# Install Spike simulator
RUN --mount=type=secret,id=GIT_ACCESS_TOKEN \
    GIT_ACCESS_TOKEN=$(cat /run/secrets/GIT_ACCESS_TOKEN) && \
    git clone https://$GIT_ACCESS_TOKEN@github.com/PSAL-POSTECH/riscv-isa-sim.git --branch TorchSim && cd riscv-isa-sim && mkdir build && cd build && \
    ../configure --prefix=$RISCV && make -j && make install && cd ../../ && rm -rf riscv-isa-sim

# Install Proxy kernel
RUN git clone https://github.com/riscv-software-src/riscv-pk.git && \
     cd riscv-pk && git checkout 4f3debe4d04f56d31089c1c716a27e2d5245e9a1 && mkdir build && cd build && \
    ../configure --prefix=$RISCV --host=riscv64-unknown-elf && make -j && make install && cd ../../ && rm -rf riscv-pk

# Prepare PyTorchSim project
COPY . /workspace/PyTorchSim

RUN cd PyTorchSim/PyTorchSimBackend && \
    mkdir -p build && \
    cd build && \
    conan install .. --build=missing && \
    cmake .. && \
    make -j$(nproc)
# Use the official Ubuntu base image
FROM ubuntu:22.04

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies including Eigen and Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    wget \
    libeigen3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install numpy matplotlib pyyaml casadi scipy

# Clone the TOGT-Planner repository
RUN git clone https://github.com/FSC-Lab/TOGT-Planner.git /opt/TOGT-Planner

# Create the cmake/eigen.cmake file inside the repository
RUN mkdir -p /opt/TOGT-Planner/cmake && \
    echo "find_package(Eigen3 3.3 REQUIRED)" > /opt/TOGT-Planner/cmake/eigen.cmake && \
    echo "include_directories(\${EIGEN3_INCLUDE_DIR})" >> /opt/TOGT-Planner/cmake/eigen.cmake

# Build the library
RUN cd /opt/TOGT-Planner && \
    mkdir build && cd build && \
    cmake .. && \
    make

# Set the working directory
WORKDIR /opt/TOGT-Planner

# Entry point to run tests or visualizations
CMD ["bash"]


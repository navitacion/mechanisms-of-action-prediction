FROM ubuntu:20.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
  cmake \
  git \
  python3 \
  python3-pip

COPY ./ ./

# Library Install By Poetry
RUN pip3 install --upgrade pip
RUN pip3 install poetry

RUN poetry config virtualenvs.create false && poetry install

# Install LightGBM
RUN git clone --recursive https://github.com/microsoft/LightGBM && cd LightGBM \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make -j4 

RUN cd LightGBM/python-package \
  && python3 setup.py install

CMD bash

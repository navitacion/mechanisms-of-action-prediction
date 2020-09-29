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

RUN echo alias pip="pip3">> ~/.bashrc \
  && echo alias python="python3">> ~/.bashrc \
  && source ~/.bashrc

# Library Install By Poetry
RUN pip install --upgrade pip
RUN pip install poetry

RUN poetry config virtualenvs.create false && poetry install

# Install LightGBM
RUN git clone --recursive https://github.com/microsoft/LightGBM && cd LightGBM \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make -j4 

RUN cd LightGBM/python-package \
  && python setup.py install

CMD bash
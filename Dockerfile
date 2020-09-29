FROM ubuntu:18.04

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
  cmake \
  git \
  python3.7 \
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
  && make -j4 \
  && cd ../python-packages \
  && python setup.py install

CMD bash
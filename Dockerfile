FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN wget https://github.com/astral-sh/uv/releases/latest/download/uv-linux-x86_64.gz -O /tmp/uv.gz \
    && gunzip /tmp/uv.gz \
    && chmod +x /tmp/uv \
    && mv /tmp/uv /usr/local/bin/

# Download the dataset
RUN wget https://cdn.orbs.cloud/data.zip -O /tmp/data.zip \
    && unzip /tmp/data.zip -d /app/data \
    && rm /tmp/data.zip

COPY pyproject.toml uv.lock ./
RUN uv sync

COPY main.py ./
COPY neuroflux_analyzer ./neuroflux_analyzer

ENTRYPOINT ["uv", "run", "main.py"]
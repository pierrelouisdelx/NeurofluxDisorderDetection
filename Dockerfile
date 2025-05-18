FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel

WORKDIR /app

RUN apt-get update && apt-get install -y wget unzip

# Install uv
RUN wget -qO- https://astral.sh/uv/install.sh | sh 

# Download the dataset
RUN wget https://cdn.orbs.cloud/data.zip -O /tmp/data.zip \
    && unzip /tmp/data.zip -d /app/data \
    && rm /tmp/data.zip

ENV PATH="/root/.local/bin/:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync

COPY main.py ./
COPY neuroflux_analyzer ./neuroflux_analyzer

ENTRYPOINT ["uv", "run", "main.py"]
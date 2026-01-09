FROM python:3.11.5-slim

WORKDIR /app

# Set the python path to include the app root
ENV PYTHONPATH="/app"

COPY requirements.txt .
RUN pip install -r requirements.txt

# Install hivemind from source
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/learning-at-home/hivemind.git && \
    cd hivemind && \
    pip install -r requirements.txt && \
    pip install .

# Copy the application code 
COPY src .

# Generate gRPC code (this ensures it's always up-to-date in the image)
RUN python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --pyi_out=. \
    --grpc_python_out=. \
    ./core/remote/nodeservice.proto


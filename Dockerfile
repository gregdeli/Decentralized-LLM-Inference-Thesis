FROM python:3.11.5-slim

WORKDIR /app

# Set the python path to include the app root
ENV PYTHONPATH "${PYTHONPATH}:/app"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code 
COPY src .
# COPY protos ./protos

# Generate gRPC code (this ensures it's always up-to-date in the image)
RUN python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --pyi_out=. \
    --grpc_python_out=. \
    ./core/remote/inference.proto


ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.06-py3
FROM ${FROM_IMAGE_NAME}

# Set working directory
WORKDIR /workspace

ENV PYTHONPATH "${PYTHONPATH}:/workspace"

COPY requirements.txt .
RUN pip install --no-cache-dir git+https://github.com/NVIDIA/dllogger.git#egg=dllogger
RUN pip install -r requirements.txt
RUN python3 -m pip install pycocotools==2.0.0

# Copy SSD code
COPY ./setup.py .
COPY ./csrc ./csrc
RUN pip install .

# Copy modified apex files to PyTorch dir
COPY ./distributed-edit.py /opt/conda/lib/python3.6/site-packages/apex/parallel/distributed.py
COPY ./hadamard.py /opt/conda/lib/python3.6/site-packages/apex/parallel/hadamard.py

COPY . .

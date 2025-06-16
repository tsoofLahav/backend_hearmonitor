# Base image with Python and PyTorch
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --upgrade typing-extensions
RUN pip install -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose port
EXPOSE 8000

# Run the server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "server:app"]

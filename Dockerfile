# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the model files for CTransformers are downloaded
RUN mkdir -p /root/.cache/transformers && \
    python -c "import ctransformers; print('ctransformers is installed correctly')"

# Copy the rest of the project files into the container
COPY . .

# Expose the default Chainlit port (8000)
EXPOSE 8000

# Command to run the Chainlit app
CMD ["chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]


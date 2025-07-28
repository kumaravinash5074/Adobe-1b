FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY document_analyzer.py .

# Ensure input and output directories exist
RUN mkdir -p /app/input /app/output

# Run the persona-driven document analyzer
CMD ["python", "document_analyzer.py"] 
# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy code
COPY app/ /app/
COPY requirements.txt /app/
COPY .env /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 7503

# Run FastAPI with uvicorn
CMD ["uvicorn", "Orchestrator_API:app", "--host", "0.0.0.0", "--port", "7503"]

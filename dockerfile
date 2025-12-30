FROM python:3.13-alpine

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install runtime deps only
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Create logs directory
RUN mkdir -p /logs

# Copy .env file
COPY .env .

# Expose the server port
EXPOSE 6901

# Start the server
CMD ["python", "app.py"]
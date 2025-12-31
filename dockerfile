# Use 3.11-slim for a smaller, faster, and more stable build
FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies
# ffmpeg (audio), libmagic (file types), and build-essential (for C-extensions)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libmagic1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy requirements
COPY requirements.txt .

# 3. Install runtime deps (No trusted-host needed usually on 3.11)
RUN pip install --no-cache-dir -r requirements.txt

# 4. Create logs and copy code
RUN mkdir -p /logs
COPY . .

# 5. Expose and Start
EXPOSE 6901
CMD ["python", "app.py"]
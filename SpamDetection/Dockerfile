# Use an official Python image as the base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your application runs on
EXPOSE 8080

# Define the command to run your application
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "SpamDetectionService:app"]


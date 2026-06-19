# Use a slim Python base image matching the project's >=3.9 constraint
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy the project files
COPY . /app

# Install the package in editable mode with test dependencies
RUN pip install --no-cache-dir -e .

# Default command runs the test suite
CMD ["pytest", "tests/", "-v"]

# Note: You can run evaluation instead of tests by passing the run command:
# docker run <image> python run_eval.py

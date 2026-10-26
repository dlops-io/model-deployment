# Use the official Python image as the base with Python 3.11 (TF 2.17 save breaks on 3.12)
FROM python:3.11-slim-bookworm

# Prevent apt from showing prompts
ENV DEBIAN_FRONTEND=noninteractive

# Python wants UTF-8 locale
ENV LANG=C.UTF-8

# Tell Python to disable buffering so we don't lose any logs.
ENV PYTHONUNBUFFERED=1

# Tell uv to copy packages from the wheel into the site-packages
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Put the virtual environment first on PATH so `python` uses it everywhere,
# including `docker exec` sessions that skip docker-entrypoint.sh
ENV VIRTUAL_ENV=/.venv
ENV PATH="/.venv/bin:$PATH"

# Install the Google Cloud CLI (gcloud) from Google's apt repository
RUN find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) \
    -exec sed -i 's|http://deb.debian.org|https://deb.debian.org|g' {} + && \
    apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates gnupg && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

# Install UV (Python package installer)
RUN pip install uv

# Set the working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies in a separate layer for better caching
RUN uv sync --frozen

# Copy the rest of the source code
COPY . ./

# Entry point
ENTRYPOINT ["/bin/bash","docker-entrypoint.sh"]

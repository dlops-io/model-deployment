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

# Install required dependencies
RUN apt-get update && \
    apt-get install -y curl apt-transport-https ca-certificates gnupg lsb-release openssh-client

# Add the Google Cloud SDK repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg


RUN find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) \
    -exec sed -i 's|http://deb.debian.org|https://deb.debian.org|g' {} + && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg lsb-release openssh-client && \
    rm -rf /var/lib/apt/lists/*    

# Install packages
# RUN apt-get update && \
#     apt-get install -y google-cloud-sdk google-cloud-sdk-gke-gcloud-auth-plugin jq docker-ce
RUN apt-get update && \
    apt-get install -y google-cloud-cli google-cloud-cli-gke-gcloud-auth-plugin jq  
# Install UV (Python package installer)
RUN pip install uv
ENV PATH="$PATH:/root/.local/bin"

# Create user
RUN groupadd -r docker && \
    useradd -ms /bin/bash app -d /home/app -u 1000 -p "$(openssl passwd -1 passw0rd)" && \
    usermod -aG docker app && \
    mkdir -p /app /.venv && \
    chown app:app /app /.venv



# Set the working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY --chown=app:app pyproject.toml uv.lock* ./

# Install dependencies in a separate layer for better caching
RUN uv sync --frozen

# Copy the rest of the source code
COPY --chown=app:app . ./

# Entry point
ENTRYPOINT ["/bin/bash","docker-entrypoint.sh"]
# Stage 1: Build stage
FROM nvcr.io/nvidia/tensorrt:23.05-py3 AS build

# Set working directory
WORKDIR /imej-denoiser

# Update apt packages
RUN apt-get update -y

# Install build dependencies
RUN apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set environment variables for uv
ENV PATH="/root/.cargo/bin:$PATH"

# Copy pyproject.toml dan file lock uv
COPY pyproject.toml uv.lock /imej-denoiser/

# Install dependencies pake uv
RUN uv sync --frozen --no-install-project --no-dev

# Stage 2: Runtime stage
FROM nvcr.io/nvidia/tensorrt:23.05-py3 AS runtime

# Set working directory
WORKDIR /imej-denoiser

# Copy dependencies dari build stage
COPY --from=build /imej-denoiser /imej-denoiser

# Set environment variables
ENV PATH="/imej-denoiser/.venv/bin:$PATH"

# Expose port yang dibutuhkan (jika perlu, misal buat aplikasi lain)
EXPOSE 8000

# Command buat tetep jalan meskipun program selesai
CMD ["tail", "-f", "/dev/null"]

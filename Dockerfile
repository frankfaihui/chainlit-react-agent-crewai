# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Set environment variables for Google Cloud Run
ENV PORT=8000
ENV HOST=0.0.0.0

# Copy project files first
COPY pyproject.toml uv.lock ./

# Install dependencies without using BuildKit mounts
RUN uv sync --frozen --no-install-project

# Copy the rest of the application
COPY . .

# Install the project
RUN uv sync --frozen

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Run the Chainlit application
CMD ["chainlit", "run", "src/chainlit_react_agent_crewai/main.py", "--port", "$PORT", "--host", "$HOST"]

# Expose the port
EXPOSE 8000 
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

ENV UV_SYSTEM_PYTHON=1 \
    UV_PYTHON=/usr/local/bin/python \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
		ffmpeg \
		python3-tk \
		coreutils \
		libsm6 \
		qtbase5-dev \
		libglu1-mesa-dev \
		libgl1-mesa-glx \
		libxcb-util1 \
		libvtk9-dev \
		procps && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

# Copy metadata first for layer caching
COPY pyproject.toml uv.lock* README.md ./

# Only install runtime dependencies
RUN uv sync --frozen --no-group dev --no-group test --no-group lint --no-install-project

# Now add source and install the project itself
COPY src ./src

RUN uv sync --locked

ENV PATH="/workspace/.venv/bin:$PATH"

CMD ["jabs-postprocess", "--help"]

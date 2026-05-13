FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.10.1 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local/ \
    PATH="/usr/local/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3.12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --python /usr/bin/python3.12 --frozen --no-install-project --no-dev

COPY . .

EXPOSE 5001
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:5001", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-", "app:application"]

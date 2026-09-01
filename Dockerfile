FROM python:3.12-slim AS runtime
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY proto ./proto
RUN pip install --no-cache-dir .
RUN useradd --create-home --uid 10001 inference
USER inference
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "llm_server.main"]


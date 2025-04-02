FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir poetry
COPY pyproject.toml poetry.lock* /app/
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-cache && \
    rm -rf /root/.cache
COPY src/ /app/
COPY config.docker.properties /app/config.properties
EXPOSE 9000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]


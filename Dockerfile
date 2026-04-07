FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8

# Зависимости ставим отдельным слоем: код меняется чаще, чем они.
COPY pyproject.toml README.md ./
# setuptools разрешает каждый пакет из pyproject уже на этапе установки, поэтому
# все объявленные каталоги должны существовать до сборки слоя с зависимостями.
# Настоящие исходники приезжают следующим слоем и перекрывают эти заглушки.
RUN mkdir -p service_classifier api data training/experiments \
 && touch service_classifier/__init__.py api/__init__.py \
          data/__init__.py training/__init__.py \
          training/experiments/__init__.py
RUN pip install --no-cache-dir .

COPY service_classifier/ service_classifier/
COPY api/ api/
COPY data/ data/
COPY training/ training/

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]

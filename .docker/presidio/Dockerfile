FROM ghcr.io/data-privacy-stack/presidio-analyzer:latest

USER root
RUN python -m spacy download da_core_news_lg
COPY conf/ /app/presidio_analyzer/conf/aarhusai/
ENV ANALYZER_CONF_FILE=presidio_analyzer/conf/aarhusai/analyzer.yaml \
    NLP_CONF_FILE=presidio_analyzer/conf/aarhusai/nlp.yaml \
    RECOGNIZER_REGISTRY_CONF_FILE=presidio_analyzer/conf/aarhusai/recognizers.yaml
USER 1001

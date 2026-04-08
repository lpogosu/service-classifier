"""Гибридный пайплайн выделения самостоятельных услуг из текста объявления."""

from service_classifier.config import Settings
from service_classifier.pipeline import Ad, PipelineResult, ServiceClassifier

__all__ = ["Ad", "PipelineResult", "ServiceClassifier", "Settings"]
__version__ = "1.0.0"

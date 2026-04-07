"""FastAPI-обёртка над пайплайном плюс веб-интерфейс для ручной проверки."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from service_classifier.config import CATEGORY_TITLES, Settings
from service_classifier.llm import OpenRouterClient, build_client
from service_classifier.pipeline import Ad, PipelineResult, ServiceClassifier

logger = logging.getLogger("service_classifier.api")

STATIC_DIR = Path(__file__).resolve().parent / "static"

#: Заполняется в lifespan; до старта приложения обращений к нему нет.
_state: dict[str, object] = {}


def get_classifier() -> ServiceClassifier:
    classifier = _state.get("classifier")
    if not isinstance(classifier, ServiceClassifier):
        raise HTTPException(status_code=503, detail="Пайплайн ещё не инициализирован")
    return classifier


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings.from_env()
    logging.basicConfig(
        level=settings.log_level, format="%(levelname)s %(name)s: %(message)s"
    )

    client = build_client(settings)
    classifier = ServiceClassifier(settings, client)
    _state["settings"] = settings
    _state["client"] = client
    _state["classifier"] = classifier

    logger.info(
        "Пайплайн готов: категорий=%d, стадия 2=%s, LLM=%s",
        len(classifier.detector.categories),
        classifier.split_mode.value,
        "online" if client is not None else "offline",
    )
    try:
        yield
    finally:
        if isinstance(client, OpenRouterClient):
            await client.close()
        _state.clear()


app = FastAPI(
    title="Service Classifier API",
    description="Выделение самостоятельных услуг из текста объявления о ремонте",
    version="1.0.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AdRequest(BaseModel):
    itemId: int | str = Field(..., description="Идентификатор объявления")
    mcId: int = Field(..., description="Исходная микрокатегория")
    mcTitle: str = Field("", description="Название исходной микрокатегории")
    description: str = Field(..., min_length=1, description="Текст объявления")


class DraftResponse(BaseModel):
    mcId: int
    mcTitle: str
    text: str
    fallback: bool = Field(False, description="Текст собран по шаблону, без LLM")


class DetectionDetail(BaseModel):
    mcId: int
    mcTitle: str
    matchedPhrases: list[str] = Field(default_factory=list)
    verification: str = Field("", description="added | removed | confirmed")


class AdResponse(BaseModel):
    detectedMcIds: list[int] = Field(default_factory=list)
    shouldSplit: bool = False
    drafts: list[DraftResponse] = Field(default_factory=list)
    detectionDetails: list[DetectionDetail] | None = None
    removedByVerification: list[DetectionDetail] | None = None


class BatchRequest(BaseModel):
    items: list[AdRequest]


class BatchResponse(BaseModel):
    results: list[AdResponse]
    processingTimeSec: float


class HealthResponse(BaseModel):
    status: str
    categoriesLoaded: int
    splitMode: str
    llm: str
    verificationModel: str
    draftModel: str


def _title(mc_id: int) -> str:
    return CATEGORY_TITLES.get(mc_id, f"Категория {mc_id}")


def _to_response(result: PipelineResult, *, details: bool) -> AdResponse:
    response = AdResponse(
        detectedMcIds=result.detected_mc_ids,
        shouldSplit=result.should_split,
        drafts=[
            DraftResponse(
                mcId=d.mc_id, mcTitle=d.mc_title, text=d.text, fallback=d.is_fallback
            )
            for d in result.drafts
        ],
    )
    if details:
        response.detectionDetails = [
            DetectionDetail(
                mcId=mc_id,
                mcTitle=_title(mc_id),
                matchedPhrases=result.matched_phrases.get(mc_id, []),
                verification=result.verification.get(mc_id, ""),
            )
            for mc_id in result.detected_mc_ids
        ]
        response.removedByVerification = [
            DetectionDetail(
                mcId=mc_id,
                mcTitle=_title(mc_id),
                matchedPhrases=phrases,
                verification="removed",
            )
            for mc_id, phrases in sorted(result.rejected.items())
        ]
    return response


@app.get("/", include_in_schema=False, response_model=None)
async def root() -> FileResponse | dict[str, str]:
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Service Classifier API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    classifier = get_classifier()
    settings = classifier.settings
    return HealthResponse(
        status="ok",
        categoriesLoaded=len(classifier.detector.categories),
        splitMode=classifier.split_mode.value,
        llm="online" if settings.llm_enabled else "offline",
        verificationModel=settings.verification_model,
        draftModel=settings.draft_model,
    )


@app.post("/process", response_model=AdResponse, response_model_exclude_none=True)
async def process_ad(request: AdRequest, details: bool = False) -> AdResponse:
    """Полный проход по четырём стадиям для одного объявления."""
    classifier = get_classifier()
    started = time.perf_counter()
    result = await classifier.process(
        Ad(
            item_id=request.itemId,
            mc_id=request.mcId,
            mc_title=request.mcTitle,
            description=request.description,
        )
    )
    logger.info(
        "[%s] detected=%s split=%s drafts=%d за %.2fс",
        request.itemId,
        result.detected_mc_ids,
        result.should_split,
        len(result.drafts),
        time.perf_counter() - started,
    )
    return _to_response(result, details=details)


@app.post("/batch", response_model=BatchResponse)
async def process_batch(request: BatchRequest) -> BatchResponse:
    classifier = get_classifier()
    started = time.perf_counter()
    results = await classifier.process_many(
        [
            Ad(
                item_id=item.itemId,
                mc_id=item.mcId,
                mc_title=item.mcTitle,
                description=item.description,
            )
            for item in request.items
        ]
    )
    return BatchResponse(
        results=[_to_response(r, details=False) for r in results],
        processingTimeSec=round(time.perf_counter() - started, 2),
    )


@app.post("/detect")
async def detect_only(request: AdRequest) -> dict[str, object]:
    """Только стадия 1 — без сети и без LLM."""
    classifier = get_classifier()
    details = classifier.detector.detect_with_details(request.description, request.mcId)
    return {
        "detectedMcIds": sorted(details),
        "details": {
            str(mc_id): {"title": _title(mc_id), "matchedPhrases": phrases}
            for mc_id, phrases in sorted(details.items())
        },
    }

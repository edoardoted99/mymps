"""FastAPI app factory + lifespan."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from mymps.server.models import ModelRegistry
from mymps.server.routes import router

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("mymps server starting up")
    app.state.registry = ModelRegistry()
    yield
    log.info("mymps server shutting down")


def create_app() -> FastAPI:
    app = FastAPI(title="mymps", lifespan=lifespan)
    app.include_router(router)
    return app

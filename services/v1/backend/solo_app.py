from __future__ import annotations

import asyncio
import contextlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .settings import env_flag, normalize_base_path, redact_internal_details
from .human_table_models import (
    HumanTableActionRequest,
    HumanTableArrangeRequest,
    HumanTableCreateRequest,
    HumanTableJoinRequest,
    HumanTableReadyRequest,
    HumanTableSession,
)
from .human_tables import EngineFactory, HumanTableError, RoomManager, TURN_SECONDS
from .models import Game
from .presentation import HandArranger


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVICE_VERSION = "1.0.0"


class SoloCreateRequest(BaseModel):
    """Public create contract for the one-human product.

    ``human_slots`` is deliberately absent from the external schema.  The
    service creates the audited internal request only after validation.
    """

    model_config = ConfigDict(extra="forbid")

    game: Game
    nickname: str = Field(min_length=1, max_length=64)
    user_id: str = Field(
        min_length=16,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
    )

    @field_validator("nickname")
    @classmethod
    def normalize_nickname(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("nickname must not be blank")
        return normalized

    @field_validator("game")
    @classmethod
    def require_guandan(cls, value: Game) -> Game:
        if value is not Game.GUANDAN:
            raise ValueError("this deployment supports guandan only")
        return value


class SoloJoinRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nickname: str = Field(min_length=1, max_length=64)
    user_id: str = Field(
        min_length=16,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    resume_token: str | None = Field(default=None, min_length=16, max_length=512)

    @field_validator("nickname")
    @classmethod
    def normalize_nickname(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("nickname must not be blank")
        return normalized


def create_solo_app(
    *,
    ui_root: Path | str | None = None,
    human_engine_factory: EngineFactory | None = None,
    human_bot_delay_ms: int | None = None,
    audit_root: Path | str | None = None,
    arranger: HandArranger | None = None,
    root_path: str | None = None,
    enable_docs: bool | None = None,
    expose_internal_details: bool | None = None,
    strict_arranger: bool | None = None,
) -> FastAPI:
    frontend_path = Path(ui_root or PROJECT_ROOT / "web").resolve()
    hand_arranger = arranger or HandArranger(frontend_path)
    if human_bot_delay_ms is None:
        try:
            human_bot_delay_ms = int(os.getenv("BATTLE_SOLO_BOT_DELAY_MS", "2000"))
        except ValueError as exc:
            raise ValueError("BATTLE_SOLO_BOT_DELAY_MS must be an integer") from exc
    resolved_audit_root = Path(
        audit_root
        if audit_root is not None
        else os.getenv(
            "BATTLE_SOLO_LOG_ROOT",
            str(PROJECT_ROOT / "data" / "solo_training_logs"),
        )
    ).expanduser()
    room_manager = RoomManager(
        engine_factory=human_engine_factory,
        arranger=hand_arranger,
        bot_delay_ms=human_bot_delay_ms,
        audit_root=resolved_audit_root,
        audit_enabled=True,
        audit_partition_by_user=True,
        replay_store=None,
    )
    resolved_root_path = normalize_base_path(
        os.getenv("BATTLE_BASE_PATH") if root_path is None else root_path
    )
    resolved_enable_docs = (
        env_flag("BATTLE_ENABLE_DOCS", False)
        if enable_docs is None
        else enable_docs
    )
    resolved_expose_internal_details = (
        env_flag("BATTLE_EXPOSE_INTERNAL_DETAILS", False)
        if expose_internal_details is None
        else expose_internal_details
    )
    resolved_strict_arranger = (
        env_flag("BATTLE_STRICT_PREFLIGHT", True)
        if strict_arranger is None
        else strict_arranger
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if resolved_strict_arranger:
            health = hand_arranger.health()
            if not health.get("ready"):
                raise RuntimeError(
                    "verified hand arranger is unavailable: "
                    + str(health.get("error") or "unknown error")
                )
        try:
            yield
        finally:
            await room_manager.close()
            hand_arranger.close()

    application = FastAPI(
        title="CardKS Solo Player Arena",
        version=SERVICE_VERSION,
        lifespan=lifespan,
        root_path=resolved_root_path,
        docs_url="/docs" if resolved_enable_docs else None,
        redoc_url="/redoc" if resolved_enable_docs else None,
        openapi_url="/openapi.json" if resolved_enable_docs else None,
    )
    application.state.room_manager = room_manager
    application.state.arranger = hand_arranger
    application.state.ui_root = frontend_path
    application.state.base_path = resolved_root_path

    def bearer_token(authorization: str | None) -> str:
        if authorization is None:
            raise HTTPException(
                status_code=401,
                detail={"code": "missing_token", "message": "missing bearer token"},
            )
        scheme, separator, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not separator or not token.strip():
            raise HTTPException(
                status_code=401,
                detail={"code": "invalid_token", "message": "invalid bearer token"},
            )
        return token.strip()

    def table_http_error(exc: HumanTableError) -> HTTPException:
        return HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.detail},
        )

    def mark_private(response: Response, *, vary_token: bool = False) -> None:
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        if vary_token:
            response.headers["Vary"] = "Authorization"

    @application.get("/api/health")
    async def health() -> dict:
        arranger_health = dict(hand_arranger.health())
        arranger_health.pop("doudizhu_mode", None)
        payload = {
            "status": "ok" if arranger_health.get("ready") else "degraded",
            "service": "cardks",
            "version": SERVICE_VERSION,
            "mode": "single-human",
            "human_slots": 1,
            "games": [Game.GUANDAN.value],
            "turn_seconds": TURN_SECONDS,
            "timeout_autoplay": False,
            "training_log": {
                "enabled": True,
                "schema": "danks_human_table_audit_v1",
                "partition": "date/user_id/table_no",
            },
            "arranger": arranger_health,
        }
        return (
            payload
            if resolved_expose_internal_details
            else redact_internal_details(payload)
        )

    @application.post(
        "/api/human-tables",
        response_model=HumanTableSession,
        status_code=201,
    )
    async def create_table(
        request: SoloCreateRequest,
        response: Response,
    ) -> HumanTableSession:
        mark_private(response)
        try:
            return await room_manager.create(
                HumanTableCreateRequest(
                    game=request.game,
                    human_slots=1,
                    nickname=request.nickname,
                    user_id=request.user_id,
                )
            )
        except HumanTableError as exc:
            raise table_http_error(exc) from exc

    @application.post(
        "/api/human-tables/{table_no}/join",
        response_model=HumanTableSession,
    )
    async def join_table(
        table_no: str,
        request: SoloJoinRequest,
        response: Response,
    ) -> HumanTableSession:
        mark_private(response)
        try:
            return await room_manager.resume_solo(
                table_no,
                HumanTableJoinRequest(
                    nickname=request.nickname,
                    user_id=request.user_id,
                    resume_token=request.resume_token,
                ),
            )
        except HumanTableError as exc:
            raise table_http_error(exc) from exc

    @application.get("/api/human-tables/{table_no}/view")
    async def view_table(
        table_no: str,
        response: Response,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict:
        mark_private(response, vary_token=True)
        try:
            return await room_manager.get_view(table_no, bearer_token(authorization))
        except HumanTableError as exc:
            raise table_http_error(exc) from exc

    @application.post("/api/human-tables/{table_no}/ready")
    async def ready_table(
        table_no: str,
        request: HumanTableReadyRequest,
        response: Response,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict:
        mark_private(response, vary_token=True)
        try:
            return await room_manager.ready(
                table_no,
                bearer_token(authorization),
                request,
            )
        except HumanTableError as exc:
            raise table_http_error(exc) from exc

    @application.post("/api/human-tables/{table_no}/actions")
    async def act_at_table(
        table_no: str,
        request: HumanTableActionRequest,
        response: Response,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict:
        mark_private(response, vary_token=True)
        try:
            return await room_manager.action(
                table_no,
                bearer_token(authorization),
                request,
            )
        except HumanTableError as exc:
            raise table_http_error(exc) from exc

    @application.post("/api/human-tables/{table_no}/arrange")
    async def arrange_hand(
        table_no: str,
        request: HumanTableArrangeRequest,
        response: Response,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict:
        mark_private(response, vary_token=True)
        try:
            return await room_manager.arrange(
                table_no,
                bearer_token(authorization),
                request,
            )
        except HumanTableError as exc:
            raise table_http_error(exc) from exc

    @application.websocket("/api/human-tables/{table_no}/ws")
    async def table_websocket(websocket: WebSocket, table_no: str) -> None:
        await websocket.accept()
        queue: asyncio.Queue[dict] | None = None
        receiver: asyncio.Task[None] | None = None
        pending_state: asyncio.Task[dict] | None = None

        async def receive_messages() -> None:
            while True:
                await websocket.receive_json()

        try:
            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=5)
            except asyncio.TimeoutError:
                await websocket.close(code=4401, reason="authentication timed out")
                return
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError, KeyError):
                await websocket.close(code=4401, reason="invalid authentication frame")
                return
            if not isinstance(message, dict) or message.get("type") != "auth":
                await websocket.close(code=4401, reason="authentication required")
                return
            token = message.get("token")
            if not isinstance(token, str) or not token:
                await websocket.close(code=4401, reason="invalid token")
                return
            try:
                queue = await room_manager.subscribe(table_no, token)
            except HumanTableError as exc:
                await websocket.close(
                    code=4404 if exc.status_code == 404 else 4401,
                    reason=exc.code,
                )
                return
            receiver = asyncio.create_task(receive_messages())
            while True:
                pending_state = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {pending_state, receiver},
                    timeout=15,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if receiver in done:
                    await receiver
                    return
                if pending_state in done:
                    await websocket.send_json(
                        {"type": "state", "state": pending_state.result()}
                    )
                    pending_state = None
                    continue
                pending_state.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pending_state
                pending_state = None
                await websocket.send_json({"type": "ping"})
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, KeyError):
            with contextlib.suppress(RuntimeError):
                await websocket.close(code=4400, reason="invalid websocket frame")
        except (WebSocketDisconnect, RuntimeError):
            return
        finally:
            if queue is not None:
                with contextlib.suppress(HumanTableError):
                    await room_manager.unsubscribe(table_no, queue)
            if pending_state is not None and not pending_state.done():
                pending_state.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pending_state
            if receiver is not None and not receiver.done():
                receiver.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receiver

    play_path = frontend_path / "play.html"
    if not play_path.is_file():
        raise RuntimeError("play.html is required by the solo frontend")
    solo_html = (
        play_path.read_text(encoding="utf-8")
        .replace("<title>Paice Arena · 人类对局</title>", "<title>CardKS · 单人对战</title>")
        .replace("<body>", '<body data-arena-mode="solo">', 1)
    )

    @application.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="./solo", status_code=307)

    @application.get("/solo", include_in_schema=False)
    async def solo_frontend() -> HTMLResponse:
        return HTMLResponse(
            solo_html,
            headers={
                "Cache-Control": "no-cache",
                "Referrer-Policy": "no-referrer",
            },
        )

    allowed_root_assets = {
        "play.js",
        "play.css",
        "styles.css",
        "kingsoft-ai-product-center-dark.svg",
    }

    @application.get("/{asset_name}", include_in_schema=False)
    async def root_asset(asset_name: str) -> FileResponse:
        if asset_name not in allowed_root_assets:
            raise HTTPException(status_code=404, detail="asset not found")
        candidate = frontend_path / asset_name
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="asset not found")
        return FileResponse(candidate)

    allowed_frontend_modules = {
        "adapters.mjs",
        "base-path.mjs",
        "card-ui.mjs",
        "hand-layout.mjs",
        "human-turn-visuals.mjs",
        "navigation.mjs",
        "room-api.mjs",
        "room-state.mjs",
        "solo-session.mjs",
        "solo-english.mjs",
    }

    @application.get("/frontend/{module_name}", include_in_schema=False)
    async def frontend_module(module_name: str) -> FileResponse:
        if module_name not in allowed_frontend_modules:
            raise HTTPException(status_code=404, detail="module not found")
        candidate = frontend_path / "frontend" / module_name
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="module not found")
        return FileResponse(candidate)

    return application


app = create_solo_app()


__all__ = ["SoloCreateRequest", "SoloJoinRequest", "app", "create_solo_app"]

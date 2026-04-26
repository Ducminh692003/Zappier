from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import time
from collections import deque
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any, AsyncGenerator
from urllib.parse import quote, urlparse
from uuid import uuid4

from curl_cffi.requests import AsyncSession
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gemini_webapi import GeminiClient, StreamSuspendedError, logger as gemini_logger
from gemini_webapi.constants import AccountStatus, Headers, Model
from gemini_webapi.types import GeneratedImage
from gemini_webapi.utils import build_cookie_fingerprint

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "local_app_static"
ENV_FILE = ROOT / ".env"
COOKIE_FILE = ROOT / "cookies.json"
IMAGE_CACHE_DIR = ROOT / "temp" / "local_app_images"
INPUT_UPLOAD_DIR = ROOT / "temp" / "local_app_inputs"
REQUIRED_COOKIE_NAMES = ("__Secure-1PSID", "__Secure-1PSIDTS")
DEFAULT_CHAT_TIMEOUT = 150.0
APP_VERSION = "0.8.6"
LOG_BUFFER: deque[dict[str, Any]] = deque(maxlen=800)
LOG_COUNTER = count(1)
AUTO_REAUTH_ABORT_TEXT = "The original request may have been silently aborted by Google."
AUTO_REAUTH_LOG_FRAGMENT = "No CID found to recover"
AUTO_REAUTH_STATUS_MESSAGE = (
    "Gemini stalled before it exposed a recoverable chat ID. "
    "Re-authenticating the session and retrying this request once."
)
MAX_AUTO_REAUTH_RETRIES = 2
STANDARD_STREAM_WATCHDOG_SECONDS = 20.0
IMAGE_STREAM_WATCHDOG_SECONDS = 45.0


def append_runtime_log(level: str, message: str) -> None:
    LOG_BUFFER.append(
        {
            "id": next(LOG_COUNTER),
            "ts": datetime.now().isoformat(timespec="seconds"),
            "level": level,
            "message": message,
        }
    )


def consume_logs(after_id: int) -> tuple[list[dict[str, Any]], int]:
    entries = [entry for entry in LOG_BUFFER if entry["id"] > after_id]
    latest_id = after_id
    if entries:
        latest_id = entries[-1]["id"]
    return entries, latest_id


def get_latest_log_id() -> int:
    return LOG_BUFFER[-1]["id"] if LOG_BUFFER else 0


def recent_runtime_logs(limit: int = 20) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    return list(LOG_BUFFER)[-limit:]


def silence_background_task(task: asyncio.Task[Any]) -> None:
    with suppress(asyncio.CancelledError, Exception):
        task.result()


def should_suppress_runtime_log(level: str, message: str) -> bool:
    if level == "DEBUG" and message.startswith("Incomplete frame at position"):
        return True
    return False


def normalize_image_quality(value: str | None) -> str:
    quality = (value or "").strip().lower()
    if quality in {"full", "fhd", "max", "preview", "source", "pending"}:
        return quality
    return "unknown"


def gemini_log_sink(message) -> None:
    record = message.record
    level = record["level"].name
    content = record["message"]
    if should_suppress_runtime_log(level, content):
        return
    append_runtime_log(level, content)


gemini_logger.add(
    gemini_log_sink,
    level="DEBUG",
    format="{message}",
    filter=lambda record: record["extra"].get("name") == "gemini_webapi",
)


def strip_wrapping_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[:1] == value[-1:] and value[:1] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ[key.strip()] = strip_wrapping_quotes(value)


def load_cookie_file(path: Path = COOKIE_FILE) -> dict[str, str]:
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    if isinstance(data, dict) and isinstance(data.get("cookies"), dict):
        return {
            str(name): str(value)
            for name, value in data["cookies"].items()
            if isinstance(name, str) and isinstance(value, str) and value
        }

    if isinstance(data, dict):
        return {
            str(name): str(value)
            for name, value in data.items()
            if isinstance(name, str) and isinstance(value, str) and value
        }

    return {}


def persist_cookie_file(cookies: dict[str, str], path: Path = COOKIE_FILE) -> None:
    payload = {"cookies": dict(sorted(cookies.items()))}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def persist_env_credentials(
    secure_1psid: str,
    secure_1psidts: str,
    path: Path = ENV_FILE,
) -> None:
    port = os.getenv("PORT", "8000").strip() or "8000"
    lines = [
        f"GEMINI_SECURE_1PSID={json.dumps(secure_1psid)}",
        f"GEMINI_SECURE_1PSIDTS={json.dumps(secure_1psidts)}",
        f"PORT={json.dumps(port)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_cookie_mapping(data: object) -> dict[str, str]:
    cookies: dict[str, str] = {}

    def add_cookie(name: object, value: object) -> None:
        if isinstance(name, str) and isinstance(value, str) and name and value:
            cookies[name] = value

    if isinstance(data, dict):
        if all(isinstance(key, str) and isinstance(value, str) for key, value in data.items()):
            for key, value in data.items():
                add_cookie(key, value)
            return cookies

        nested = data.get("cookies")
        if isinstance(nested, dict):
            return extract_cookie_mapping(nested)
        if isinstance(nested, list):
            return extract_cookie_mapping(nested)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                add_cookie(item.get("name"), item.get("value"))

    return cookies


def parse_cookie_blob(raw_text: str) -> dict[str, str]:
    text = raw_text.strip()
    if not text:
        raise ValueError("Paste your exported cookie rows or a cookies JSON payload.")

    cookies: dict[str, str] = {}

    if text[:1] in "{[":
        try:
            cookies.update(extract_cookie_mapping(json.loads(text)))
        except json.JSONDecodeError:
            pass

    if not cookies:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "\t" in raw_line:
                parts = [strip_wrapping_quotes(part.strip()) for part in raw_line.split("\t")]
                if len(parts) >= 2 and parts[0] and parts[1]:
                    cookies[parts[0]] = parts[1]
                    continue

            if "=" in line:
                name, value = line.split("=", 1)
                name = name.strip()
                value = strip_wrapping_quotes(value.strip())
                if name and value:
                    cookies[name] = value
                    continue

            parts = line.split()
            if len(parts) >= 2 and parts[0] and parts[1]:
                cookies[parts[0]] = strip_wrapping_quotes(parts[1])

    missing = [name for name in REQUIRED_COOKIE_NAMES if not cookies.get(name)]
    if missing:
        raise ValueError(
            "Missing required cookies: "
            + ", ".join(missing)
            + ". Paste the raw export that includes both __Secure-1PSID and __Secure-1PSIDTS."
        )

    return cookies


def load_configured_cookies() -> tuple[dict[str, str], str]:
    load_env_file()

    cookies = load_cookie_file()
    sources: list[str] = []
    if cookies:
        sources.append("cookies.json")

    env_mapping = {
        "__Secure-1PSID": os.getenv("GEMINI_SECURE_1PSID", "").strip(),
        "__Secure-1PSIDTS": os.getenv("GEMINI_SECURE_1PSIDTS", "").strip(),
    }
    env_used = False
    for name, value in env_mapping.items():
        if value and not cookies.get(name):
            cookies[name] = value
            env_used = True

    if env_used or any(env_mapping.values()):
        sources.append(".env")

    return cookies, " + ".join(dict.fromkeys(sources)) or "none"


def mask_cookie(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 12:
        return value
    return f"{value[:8]}...{value[-6:]}"


def sanitize_download_name(filename: str | None, fallback: str = "gemini-image.png") -> str:
    if not filename:
        return fallback
    safe_name = Path(filename).name.strip()
    return safe_name or fallback


def guess_image_extension(mime_type: str | None) -> str:
    normalized = (mime_type or "").strip().lower()
    return {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/heic": ".heic",
    }.get(normalized, ".png")


def decode_image_input(raw_value: str) -> tuple[bytes, str]:
    text = raw_value.strip()
    if not text:
        raise ValueError("Image payload cannot be empty.")

    payload = text
    extension = ".png"

    if text.startswith("data:"):
        header, _, encoded = text.partition(",")
        if not encoded:
            raise ValueError("Image data URL is missing the base64 payload.")
        payload = encoded.strip()
        mime_type = header[5:].split(";", 1)[0]
        extension = guess_image_extension(mime_type)

    try:
        decoded = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Could not decode one of the image inputs as base64.") from exc

    if not decoded:
        raise ValueError("Decoded image payload is empty.")

    return decoded, extension


def summarize_credentials(cookies: dict[str, str], source: str) -> dict[str, object]:
    return {
        "configured": bool(cookies.get("__Secure-1PSID")),
        "cookieCount": len(cookies),
        "source": source,
        "fingerprint": build_cookie_fingerprint(cookies),
        "masked": {
            "__Secure-1PSID": mask_cookie(cookies.get("__Secure-1PSID")),
            "__Secure-1PSIDTS": mask_cookie(cookies.get("__Secure-1PSIDTS")),
        },
    }


def summarize_account_status(status: AccountStatus | None) -> dict[str, str] | None:
    if status is None:
        return None

    level = "info"
    if status != AccountStatus.AVAILABLE:
        level = "warning"

    return {
        "code": str(int(status)),
        "name": status.name,
        "description": status.description,
        "level": level,
    }


def build_history_state(
    *,
    cid: str,
    status: str,
    message: str,
    checked: bool,
    saved: bool,
) -> dict[str, Any]:
    return {
        "status": status,
        "saved": saved,
        "checked": checked,
        "message": message,
        "chatId": cid or None,
        "chatUrl": f"https://gemini.google.com/app/{cid}" if cid else None,
    }


def summarize_active_auth(
    client: GeminiClient | None,
    configured_fingerprint: str | None,
) -> dict[str, Any] | None:
    if client is None:
        return None

    active_configured_fingerprint = getattr(
        client, "configured_cookie_fingerprint", None
    )
    matches_configured = (
        bool(configured_fingerprint)
        and bool(active_configured_fingerprint)
        and configured_fingerprint == active_configured_fingerprint
    )

    return {
        "authSource": getattr(client, "auth_source", None),
        "authCookieFingerprint": getattr(client, "auth_cookie_fingerprint", None),
        "configuredCookieFingerprint": active_configured_fingerprint,
        "matchesConfigured": matches_configured,
        "accountStatus": summarize_account_status(getattr(client, "account_status", None)),
    }


def looks_like_image_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(
        token in lowered
        for token in (
            "generate image",
            "generate an image",
            "create image",
            "create an image",
            "draw ",
            "render ",
            "illustration",
            "picture",
            "photo",
            "suncatcher",
            "cat suncatcher",
            "image of",
            "tạo ảnh",
            "tạo hình",
            "tạo hình ảnh",
            "vẽ ",
            "ảnh ",
            "hình ảnh",
        )
    )


def normalize_preview_size(value: int | None) -> int:
    try:
        size = int(value or 0)
    except (TypeError, ValueError):
        size = 0
    return 4096 if size >= 4096 else 2048


def preferred_generated_preview_size(use_pro: bool) -> int:
    return 4096 if use_pro else 2048


def resolve_image_mode_label(prompt_looks_like_image: bool, use_pro: bool) -> str | None:
    if not prompt_looks_like_image:
        return None
    return "Image generation Pro" if use_pro else "Image generation"


def resolve_stream_watchdog_timeout(prompt_looks_like_image: bool) -> float:
    return (
        IMAGE_STREAM_WATCHDOG_SECONDS
        if prompt_looks_like_image
        else STANDARD_STREAM_WATCHDOG_SECONDS
    )


def model_entry_is_pro(entry: dict[str, str]) -> bool:
    name = str(entry.get("name") or "").lower()
    label = str(entry.get("label") or "").lower()
    return (
        label == "pro"
        or name.startswith("gemini-3-pro")
        or name.startswith("gemini-3.1-pro")
        or name.startswith("gemini-3.0-pro")
        or name in {
            Model.ADVANCED_PRO.model_name,
            Model.PLUS_PRO.model_name,
            Model.BASIC_PRO.model_name,
        }
    )


def resolve_nano_banana_pro_model(
    available_models: list[dict[str, str]],
) -> str | None:
    if not available_models:
        return Model.ADVANCED_PRO.model_name

    candidates = [entry for entry in available_models if model_entry_is_pro(entry)]
    if not candidates:
        return None

    def priority(entry: dict[str, str]) -> tuple[int, str]:
        name = str(entry.get("name") or "").lower()
        label = str(entry.get("label") or "").lower()
        if label == "pro" and "advanced" in name:
            return (0, name)
        if "advanced" in name:
            return (1, name)
        if "plus" in name:
            return (2, name)
        if label == "pro":
            return (3, name)
        return (4, name)

    chosen = sorted(candidates, key=priority)[0]
    return str(chosen.get("name") or "").strip() or None


def should_use_pro_image_path(
    *,
    prompt: str,
    requested_model: str,
    explicit_use_pro: bool,
) -> bool:
    if explicit_use_pro:
        return True
    return looks_like_image_prompt(prompt) and model_entry_is_pro(
        {"name": requested_model, "label": requested_model}
    )


def output_has_images(output: Any | None) -> bool:
    return bool(list(getattr(output, "images", None) or []))


def should_retry_standard_image_generation(
    *,
    prompt_looks_like_image: bool,
    requested_use_pro: bool,
    active_use_pro: bool,
    fallback_attempted: bool,
    output: Any | None,
) -> bool:
    # A checked Nano Banana Pro request must not silently degrade to the normal
    # image path, otherwise the UI can show a successful image from the wrong model.
    return False


def normalize_request_source(source: str | None) -> str:
    value = (source or "").strip()
    return value or "local-ui"


def infer_request_source(request: Request | None) -> str:
    if request is None:
        return "local-ui"

    headers = request.headers
    candidates = [
        headers.get("x-client-source", ""),
        headers.get("x-client-name", ""),
        headers.get("x-forwarded-host", ""),
        headers.get("origin", ""),
        headers.get("referer", ""),
        headers.get("user-agent", ""),
    ]
    combined = " ".join(part.lower() for part in candidates if part)
    if "higgsflow" in combined or "higgs flow" in combined or "higgs" in combined:
        return "HiggsFlow"

    for raw_url in (headers.get("origin", ""), headers.get("referer", "")):
        try:
            host = urlparse(raw_url).netloc.lower()
        except Exception:
            host = ""
        if host in {"127.0.0.1:8000", "localhost:8000"}:
            return "local-ui"

    user_agent = headers.get("user-agent", "").lower()
    if request.client and request.client.host in {"127.0.0.1", "::1"} and "mozilla" in user_agent:
        return "local-ui"

    return "api-client"


class ChatRequest(BaseModel):
    prompt: str
    model: str = Model.UNSPECIFIED.model_name
    reset: bool = False
    temporary: bool = False
    use_pro: bool = False
    timeout_seconds: float = Field(default=DEFAULT_CHAT_TIMEOUT, ge=15, le=600)
    images: list[str] = Field(default_factory=list)


class ResetRequest(BaseModel):
    model: str = Model.UNSPECIFIED.model_name


class CookieUpdateRequest(BaseModel):
    raw_cookies: str
    persist: bool = True


@dataclass
class GeminiLocalService:
    client: GeminiClient | None = None
    chat: object | None = None
    current_model: str = Model.UNSPECIFIED.model_name
    available_models: list[dict[str, str]] = field(default_factory=list)
    cookie_values: dict[str, str] = field(default_factory=dict)
    cookie_source: str = "none"
    configured_cookie_fingerprint: str | None = None
    runtime_cookie_values: dict[str, str] = field(default_factory=dict)
    runtime_cookie_source: str | None = None
    account_status: AccountStatus | None = None
    boot_error: str | None = None
    generated_image_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    generated_image_tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)
    _init_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _chat_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _session_state: asyncio.Condition = field(default_factory=asyncio.Condition)
    _active_session_operations: dict[str, dict[str, Any]] = field(default_factory=dict)
    _reprobe_in_progress: bool = False
    _session_generation: int = 0

    def _sync_cookie_snapshot(self, cookies: dict[str, str], source: str) -> None:
        self.cookie_values = dict(cookies)
        self.cookie_source = source
        self.configured_cookie_fingerprint = build_cookie_fingerprint(cookies)

    def _resolve_configured_cookies(self) -> tuple[dict[str, str], str]:
        if self.runtime_cookie_values:
            return dict(self.runtime_cookie_values), self.runtime_cookie_source or "runtime paste"
        return load_configured_cookies()

    def _build_status_payload(
        self,
        *,
        ready: bool,
        error: str | None,
        recent_logs: list[dict[str, Any]],
        reprobed: bool = False,
    ) -> dict[str, object]:
        active_auth = summarize_active_auth(self.client, self.configured_cookie_fingerprint)
        busy_operations = self._session_activity_snapshot()
        return {
            "ready": ready,
            "version": APP_VERSION,
            "currentModel": self.current_model,
            "models": self.available_models,
            "error": error,
            "reprobed": reprobed,
            "credentials": summarize_credentials(self.cookie_values, self.cookie_source),
            "accountStatus": summarize_account_status(self.account_status),
            "activeAuth": active_auth,
            "recentLogs": recent_logs[-12:],
            "sessionBusy": bool(busy_operations) or self._reprobe_in_progress,
            "busyOperations": busy_operations,
            "busyMessage": self._session_busy_message(
                "Gemini session refresh",
                busy_operations,
            )
            if busy_operations
            else (
                "Gemini session refresh is already running."
                if self._reprobe_in_progress
                else None
            ),
        }

    def _session_activity_snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "source": entry.get("source"),
                "operation": entry.get("operation"),
            }
            for entry in self._active_session_operations.values()
        ]

    def _session_busy_message(
        self,
        action: str,
        busy_operations: list[dict[str, Any]] | None = None,
    ) -> str:
        entries = busy_operations if busy_operations is not None else self._session_activity_snapshot()
        if not entries:
            return f"{action} is waiting for Gemini to go idle."

        labels = [
            f"{normalize_request_source(entry.get('source'))} {entry.get('operation') or 'session activity'}"
            for entry in entries[:3]
        ]
        suffix = f" (+{len(entries) - 3} more)" if len(entries) > 3 else ""
        return f"{action} is paused because Gemini is still busy with " + ", ".join(labels) + suffix + "."

    @asynccontextmanager
    async def _hold_session_activity(
        self,
        operation: str,
        source: str | None = None,
    ):
        activity_id = uuid4().hex[:8]
        entry = {
            "source": normalize_request_source(source),
            "operation": operation,
            "startedAt": time.monotonic(),
        }
        async with self._session_state:
            while self._reprobe_in_progress:
                await self._session_state.wait()
            self._active_session_operations[activity_id] = entry

        try:
            yield entry
        finally:
            async with self._session_state:
                self._active_session_operations.pop(activity_id, None)
                self._session_state.notify_all()

    async def _initialize_locked(self, cookies: dict[str, str], source: str) -> None:
        secure_1psid = cookies.get("__Secure-1PSID", "").strip()
        secure_1psidts = cookies.get("__Secure-1PSIDTS", "").strip()
        if not secure_1psid:
            raise RuntimeError(
                "Missing GEMINI credentials. Paste raw cookies or set GEMINI_SECURE_1PSID in .env."
            )

        client = GeminiClient(
            secure_1psid=secure_1psid,
            secure_1psidts=secure_1psidts,
        )

        extra_cookies = {
            name: value
            for name, value in cookies.items()
            if name not in REQUIRED_COOKIE_NAMES and value
        }
        if extra_cookies:
            client.cookies = extra_cookies

        try:
            await client.init(
                timeout=max(120, int(DEFAULT_CHAT_TIMEOUT)),
                auto_close=False,
                auto_refresh=False,
                verbose=True,
            )
        except Exception:
            await client.close()
            raise

        discovered_models = [
            {
                "name": model.model_name,
                "label": getattr(model, "display_name", model.model_name),
            }
            for model in (client.list_models() or [])
        ] or [{"name": model.model_name, "label": model.model_name} for model in Model]

        allowed_models = {item["name"] for item in discovered_models}
        if self.current_model not in allowed_models:
            self.current_model = discovered_models[0]["name"]

        self.client = client
        self._session_generation += 1
        setattr(client, "session_generation", self._session_generation)
        self._sync_cookie_snapshot(cookies, source)
        self.account_status = getattr(client, "account_status", None)
        self.available_models = discovered_models
        self.chat = client.start_chat(model=self.current_model)
        self.boot_error = None

    async def reprobe_auth(self, force: bool = False) -> bool:
        cookies, source = self._resolve_configured_cookies()
        configured_fingerprint = build_cookie_fingerprint(cookies)

        active_fingerprint = (
            getattr(self.client, "configured_cookie_fingerprint", None)
            if self.client is not None
            else None
        )
        needs_reprobe = force or self.client is None or configured_fingerprint != active_fingerprint
        self._sync_cookie_snapshot(cookies, source)

        if not needs_reprobe:
            return False

        async with self._session_state:
            busy_operations = self._session_activity_snapshot()
            if busy_operations:
                if force:
                    append_runtime_log(
                        "INFO",
                        self._session_busy_message("Gemini re-auth", busy_operations),
                    )
                return False
            if self._reprobe_in_progress:
                return False
            self._reprobe_in_progress = True

        try:
            async with self._init_lock:
                cookies, source = self._resolve_configured_cookies()
                configured_fingerprint = build_cookie_fingerprint(cookies)
                active_fingerprint = (
                    getattr(self.client, "configured_cookie_fingerprint", None)
                    if self.client is not None
                    else None
                )
                needs_reprobe = (
                    force or self.client is None or configured_fingerprint != active_fingerprint
                )
                self._sync_cookie_snapshot(cookies, source)
                if not needs_reprobe:
                    return False

                async with self._chat_lock:
                    await self.close()
                    await self._initialize_locked(cookies, source)

                append_runtime_log(
                    "INFO",
                    "Re-probed Gemini auth using the latest configured cookies.",
                )
                return True
        finally:
            async with self._session_state:
                self._reprobe_in_progress = False
                self._session_state.notify_all()

    async def ensure_ready(self) -> None:
        if self.client is not None:
            return

        async with self._init_lock:
            if self.client is not None:
                return

            cookies, source = self._resolve_configured_cookies()
            self._sync_cookie_snapshot(cookies, source)
            await self._initialize_locked(cookies, source)

    async def close(self) -> None:
        tasks = list(self.generated_image_tasks.values())
        self.generated_image_tasks = {}
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await task

        client = self.client
        self.client = None
        self.chat = None
        self.available_models = []
        self.account_status = None
        if client is not None:
            await client.close()

    async def status(self) -> dict[str, object]:
        try:
            reprobed = await self.reprobe_auth(force=False)
            logs, _ = consume_logs(max(get_latest_log_id() - 12, 0))
            return self._build_status_payload(
                ready=True,
                error=None,
                recent_logs=logs,
                reprobed=reprobed,
            )
        except Exception as exc:
            self.boot_error = str(exc)
            logs, _ = consume_logs(max(get_latest_log_id() - 12, 0))
            return self._build_status_payload(
                ready=False,
                error=str(exc),
                recent_logs=logs,
            )

    async def update_credentials(self, payload: CookieUpdateRequest) -> dict[str, object]:
        cookies = parse_cookie_blob(payload.raw_cookies)
        source = "cookies.json + .env" if payload.persist else "runtime paste"
        next_runtime_cookie_values = {} if payload.persist else dict(cookies)
        next_runtime_cookie_source = None if payload.persist else source

        busy_operations = self._session_activity_snapshot()
        if busy_operations or self._reprobe_in_progress:
            raise RuntimeError(
                self._session_busy_message("Applying fresh cookies", busy_operations)
                if busy_operations
                else "Applying fresh cookies is paused because Gemini is re-authenticating right now."
            )

        async with self._init_lock:
            async with self._chat_lock:
                await self.close()
                await self._initialize_locked(
                    cookies,
                    source,
                )

        self.runtime_cookie_values = next_runtime_cookie_values
        self.runtime_cookie_source = next_runtime_cookie_source

        os.environ["GEMINI_SECURE_1PSID"] = cookies["__Secure-1PSID"]
        os.environ["GEMINI_SECURE_1PSIDTS"] = cookies.get("__Secure-1PSIDTS", "")

        if payload.persist:
            persist_cookie_file(cookies)
            persist_env_credentials(
                cookies["__Secure-1PSID"],
                cookies.get("__Secure-1PSIDTS", ""),
            )
            self.cookie_source = source

        append_runtime_log(
            "INFO",
            "Fresh cookies parsed and Gemini client reinitialized from local app.",
        )
        logs, _ = consume_logs(max(get_latest_log_id() - 12, 0))
        payload_data = self._build_status_payload(
            ready=True,
            error=None,
            recent_logs=logs,
            reprobed=True,
        )
        payload_data["ok"] = True
        return payload_data

    async def reset(self, model_name: str) -> dict[str, object]:
        await self.ensure_ready()
        assert self.client is not None

        self.current_model = model_name or Model.UNSPECIFIED.model_name
        self.chat = self.client.start_chat(model=self.current_model)
        append_runtime_log("INFO", f"Chat session reset for model {self.current_model}.")
        return {
            "ok": True,
            "currentModel": self.current_model,
        }

    def _resolve_effective_model(self, requested_model: str, *, use_pro: bool) -> str:
        requested = requested_model or Model.UNSPECIFIED.model_name
        if not use_pro:
            return requested

        pro_model = resolve_nano_banana_pro_model(self.available_models)
        if not pro_model:
            available = ", ".join(
                f"{item.get('label') or item.get('name')} ({item.get('name')})"
                for item in self.available_models
            )
            raise RuntimeError(
                "Nano Banana Pro is selected, but this Gemini session did not expose a Pro model via list_models(). "
                f"Available models: {available or 'none'}"
            )

        if requested != Model.UNSPECIFIED.model_name and model_entry_is_pro(
            {"name": requested, "label": requested}
        ):
            return requested

        if requested != pro_model:
            append_runtime_log(
                "INFO",
                "Nano Banana Pro selected. Using "
                f"{pro_model} instead of {requested} so Gemini receives the Pro image request on a Pro-capable model.",
            )
        return pro_model

    def _should_auto_reprobe_stream_error(
        self,
        exc: Exception,
        *,
        latest_output: Any | None = None,
        partial_text: str = "",
        partial_thoughts: str = "",
    ) -> bool:
        if latest_output is not None or partial_text or partial_thoughts:
            return False

        if isinstance(exc, StreamSuspendedError):
            return True

        message = str(exc or "")
        if AUTO_REAUTH_LOG_FRAGMENT in message or "Stream suspended (" in message:
            return True

        if AUTO_REAUTH_ABORT_TEXT not in message:
            return False

        for entry in recent_runtime_logs(24):
            content = entry.get("message", "")
            if AUTO_REAUTH_LOG_FRAGMENT in content or "Stream suspended (" in content:
                return True

        return False

    def _restart_chat_for_retry_locked(self) -> None:
        assert self.client is not None
        self.chat = self.client.start_chat(model=self.current_model)
        append_runtime_log(
            "INFO",
            f"Started a fresh chat session for retry on model {self.current_model}.",
        )

    def _materialize_input_files(
        self,
        images: list[str],
        request_tag: str,
    ) -> tuple[list[str], Path | None]:
        if not images:
            return [], None

        target_dir = INPUT_UPLOAD_DIR / request_tag
        target_dir.mkdir(parents=True, exist_ok=True)
        file_paths: list[str] = []

        for index, image in enumerate(images):
            image_bytes, extension = decode_image_input(image)
            file_path = target_dir / f"input_{index}{extension}"
            file_path.write_bytes(image_bytes)
            file_paths.append(str(file_path))

        append_runtime_log(
            "INFO",
            f"Prepared {len(file_paths)} local input image(s) for Gemini request {request_tag}.",
        )
        return file_paths, target_dir

    def _cleanup_input_files(self, target_dir: Path | None) -> None:
        if target_dir is None or not target_dir.exists():
            return

        for child in target_dir.iterdir():
            if child.is_file():
                try:
                    child.unlink()
                except OSError:
                    pass

        try:
            target_dir.rmdir()
        except OSError:
            pass

    def _register_generated_image(
        self,
        image: GeneratedImage,
        request_tag: str,
        index: int,
        request_source: str = "local-ui",
        use_pro: bool = False,
        image_mode: str | None = None,
    ) -> dict[str, Any]:
        token = uuid4().hex
        record = {
            "token": token,
            "requestTag": request_tag,
            "requestSource": normalize_request_source(request_source),
            "sessionGeneration": self._session_generation,
            "index": index,
            "title": image.title,
            "alt": image.alt,
            "previewUrl": getattr(image, "preview_url", "") or image.url,
            "sourceUrl": image.url,
            "cid": image.cid,
            "rid": image.rid,
            "rcid": image.rcid,
            "imageId": image.image_id,
            "previewRelativePath": None,
            "cachedRelativePath": None,
            "browserUrl": None,
            "usePro": use_pro,
            "preferredPreviewSize": preferred_generated_preview_size(use_pro),
            "imageMode": image_mode,
            "downloadName": sanitize_download_name(
                f"{Path(image.alt or image.title or f'image_{index + 1}').stem}.png"
            ),
            "quality": "pending",
            "cacheStatus": "pending",
            "error": None,
        }
        self.generated_image_records[token] = record
        return record

    def _generated_image_cache_status(self, record: dict[str, Any]) -> str:
        token = record["token"]
        if self._generated_image_file_path(record) is not None:
            return "ready"
        if record.get("error"):
            return "failed"
        task = self.generated_image_tasks.get(token)
        if task is not None and not task.done():
            return "caching"
        return record.get("cacheStatus") or "pending"

    def _generated_image_file_path(self, record: dict[str, Any]) -> Path | None:
        relative_path = record.get("cachedRelativePath")
        if not relative_path:
            return None
        candidate = IMAGE_CACHE_DIR / relative_path
        return candidate if candidate.exists() else None

    def _generated_image_preview_path(self, record: dict[str, Any]) -> Path | None:
        relative_path = record.get("previewRelativePath")
        if not relative_path:
            return None
        candidate = IMAGE_CACHE_DIR / relative_path
        return candidate if candidate.exists() else None

    @staticmethod
    def _infer_generated_browser_quality(url: str | None) -> str:
        value = (url or "").lower()
        if "=s4096" in value:
            return "max"
        if "=s2048" in value:
            return "fhd"
        if "d-i?alr=yes" in value:
            return "preview"
        return "preview"

    @staticmethod
    def _build_generated_browser_candidates(url: str, preview_size: int = 2048) -> list[str]:
        candidates: list[str] = []

        def add(candidate: str | None) -> None:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        add(GeneratedImage._build_scaled_candidate(url, normalize_preview_size(preview_size)))
        add(url)
        add(GeneratedImage._build_handoff_candidate(url))
        return candidates

    @staticmethod
    def _normalize_generated_browser_url(url: str, preview_size: int = 2048) -> str:
        upgraded = GeneratedImage._build_scaled_candidate(
            url,
            normalize_preview_size(preview_size),
        )
        return upgraded or url

    async def _resolve_generated_browser_url(self, record: dict[str, Any]) -> str | None:
        preview_size = normalize_preview_size(record.get("preferredPreviewSize"))
        for candidate in (
            record.get("browserUrl"),
            record.get("previewUrl"),
            record.get("sourceUrl"),
        ):
            normalized = self._normalize_generated_browser_url(
                candidate or "",
                preview_size=preview_size,
            )
            if normalized.startswith("http"):
                record["browserUrl"] = normalized
                return normalized
        return None

    def _build_generated_image_payload(self, record: dict[str, Any]) -> dict[str, Any]:
        token = record["token"]
        base_url = f"/api/generated-image/{token}"
        cached_path = self._generated_image_file_path(record)
        preview_path = self._generated_image_preview_path(record)
        cached = cached_path is not None
        cache_status = self._generated_image_cache_status(record)
        browser_url = record.get("browserUrl")
        local_preview_ready = cached or preview_path is not None
        remote_preview_ready = bool(
            browser_url or record.get("previewUrl") or record.get("sourceUrl")
        )
        preview_ready = local_preview_ready or remote_preview_ready
        url_version = normalize_image_quality(record.get("quality")) if cached else cache_status
        proxy_url = f"{base_url}?v={url_version}" if preview_ready else None
        download_status = "ready" if cached else ("failed" if cache_status == "failed" else "pending")
        return {
            "title": record["title"],
            "alt": record["alt"],
            "sourceUrl": record.get("previewUrl") or record.get("sourceUrl"),
            "browserUrl": browser_url,
            "proxyUrl": proxy_url,
            "downloadUrl": f"{base_url}?download=true" if cached else None,
            "downloadName": record.get("downloadName"),
            "quality": normalize_image_quality(record.get("quality")),
            "cached": cached,
            "downloadReady": cached,
            "previewReady": preview_ready,
            "localPreviewReady": local_preview_ready,
            "cacheStatus": cache_status,
            "downloadStatus": download_status,
            "token": token,
            "serverManaged": True,
            "imageMode": record.get("imageMode"),
            "error": record.get("error"),
            "saveError": record.get("error"),
        }

    def _build_generated_image_from_record(self, record: dict[str, Any]) -> GeneratedImage:
        assert self.client is not None
        return GeneratedImage(
            url=record.get("previewUrl") or record.get("sourceUrl") or "",
            preview_url=record.get("previewUrl") or record.get("sourceUrl") or "",
            title=record.get("title") or "[Generated Image]",
            alt=record.get("alt") or "",
            proxy=self.client.proxy,
            client=self.client.client,
            client_ref=self.client,
            cid=record.get("cid") or "",
            rid=record.get("rid") or "",
            rcid=record.get("rcid") or "",
            image_id=record.get("imageId") or "",
            preferred_preview_size=normalize_preview_size(record.get("preferredPreviewSize")),
        )

    async def _cache_generated_image_record(self, token: str) -> dict[str, Any]:
        await self.ensure_ready()
        assert self.client is not None

        record = self.generated_image_records.get(token)
        if record is None:
            raise KeyError(f"Unknown generated image token: {token}")

        existing_path = self._generated_image_file_path(record)
        if existing_path is not None:
            record["cacheStatus"] = "ready"
            return record

        image = self._build_generated_image_from_record(record)
        target_dir = IMAGE_CACHE_DIR / record["requestTag"]
        target_dir.mkdir(parents=True, exist_ok=True)
        async with self._hold_session_activity(
            f"generated image save ({record['requestTag']} #{record['index'] + 1})",
            record.get("requestSource"),
        ):
            saved_path = await image.save(
                path=str(target_dir),
                filename=f"image_{record['index']}",
                verbose=False,
                full_size=True,
            )

        relative_path = Path(saved_path).relative_to(IMAGE_CACHE_DIR).as_posix()
        record["cachedRelativePath"] = relative_path
        record["downloadName"] = Path(saved_path).name
        record["quality"] = normalize_image_quality(getattr(image, "saved_quality", None))
        record["sourceUrl"] = image.url
        record["previewUrl"] = getattr(image, "preview_url", "") or record.get("previewUrl")
        record["cacheStatus"] = "ready"
        record["error"] = None
        append_runtime_log(
            "INFO",
            f"Saved Gemini generated image {record['index'] + 1} via authenticated fetch ({record['quality']}).",
        )
        return record

    async def _cache_generated_image_preview_record(self, token: str) -> dict[str, Any]:
        await self.ensure_ready()
        assert self.client is not None

        record = self.generated_image_records.get(token)
        if record is None:
            raise KeyError(f"Unknown generated image token: {token}")

        existing_preview = self._generated_image_preview_path(record)
        if existing_preview is not None or self._generated_image_file_path(record) is not None:
            return record

        browser_url = await self._resolve_generated_browser_url(record)
        image = self._build_generated_image_from_record(record)
        if browser_url:
            image.url = browser_url
            image.preview_url = browser_url

        target_dir = IMAGE_CACHE_DIR / record["requestTag"]
        target_dir.mkdir(parents=True, exist_ok=True)

        async with self._hold_session_activity(
            f"generated image preview ({record['requestTag']} #{record['index'] + 1})",
            record.get("requestSource"),
        ):
            saved_path = await image.save(
                path=str(target_dir),
                filename=f"image_{record['index']}_preview",
                verbose=False,
                full_size=False,
            )

        relative_path = Path(saved_path).relative_to(IMAGE_CACHE_DIR).as_posix()
        record["previewRelativePath"] = relative_path
        record["previewUrl"] = getattr(image, "preview_url", "") or record.get("previewUrl")
        record["browserUrl"] = browser_url or record.get("browserUrl")
        return record

    def _start_generated_image_cache(self, token: str, delay_seconds: float = 0.0) -> None:
        existing = self.generated_image_tasks.get(token)
        if existing is not None and not existing.done():
            return

        record = self.generated_image_records.get(token)
        if record is None:
            return
        record["cacheStatus"] = "caching"

        async def runner() -> None:
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
            try:
                try:
                    preview_before = self._generated_image_preview_path(record)
                    await self._cache_generated_image_preview_record(token)
                    if preview_before is None and self._generated_image_preview_path(record) is not None:
                        append_runtime_log(
                            "INFO",
                            f"Rendered Gemini image {record['index'] + 1} locally for preview.",
                        )
                except Exception as exc:
                    append_runtime_log(
                        "DEBUG",
                        f"Preview cache for Gemini generated image {record['index'] + 1} was not ready yet. {exc}",
                    )
                await self._cache_generated_image_record(token)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._resolve_generated_browser_url(record)
                record["error"] = str(exc)
                record["cacheStatus"] = "failed"
                append_runtime_log(
                    "WARNING",
                    "Background local cache for Gemini generated image "
                    f"{record['index'] + 1} failed. Direct preview is still available. {exc}",
                )
            finally:
                current = self.generated_image_tasks.get(token)
                if current is task:
                    self.generated_image_tasks.pop(token, None)

        task = asyncio.create_task(
            runner(),
            name=f"generated-image-cache-{token[:8]}",
        )
        self.generated_image_tasks[token] = task

    async def get_generated_image_status(self, token: str) -> dict[str, object]:
        record = self.generated_image_records.get(token)
        if record is None:
            raise HTTPException(status_code=404, detail="Generated image token was not found.")

        if (
            self._generated_image_file_path(record) is None
            and not record.get("error")
            and token not in self.generated_image_tasks
        ):
            self._start_generated_image_cache(token)

        return {
            "ok": True,
            "image": self._build_generated_image_payload(record),
        }

    async def _fetch_image_proxy_response(
        self,
        url: str,
        *,
        download_name: str | None = None,
        as_download: bool = False,
        request_source: str = "local-ui",
    ) -> Response:
        await self.ensure_ready()
        assert self.client is not None and self.client.client is not None

        current_url = url
        response = None
        request_headers = dict(Headers.REFERER.value)

        async with self._hold_session_activity("image proxy fetch", request_source):
            for _ in range(8):
                response = await self.client.client.get(current_url, headers=request_headers)
                if response.status_code != 200:
                    break

                content_type = (
                    response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                )
                if content_type.startswith("image/"):
                    headers = {"Cache-Control": "no-store"}
                    if as_download:
                        headers["Content-Disposition"] = (
                            f'attachment; filename="{sanitize_download_name(download_name)}"'
                        )
                    return Response(
                        content=response.content,
                        media_type=content_type or "image/png",
                        headers=headers,
                    )

                if content_type.startswith("text/plain"):
                    next_url = response.text.strip()
                    if next_url.startswith("http"):
                        current_url = next_url
                        continue

                break

        raise HTTPException(
            status_code=502,
            detail=(
                f"Image proxy failed: HTTP Error {response.status_code if response else 'N/A'}: "
                f"{getattr(response, 'reason', '') if response else ''}"
            ).rstrip(),
        )

    async def get_generated_image_response(
        self,
        token: str,
        download: bool = False,
        request_source: str = "local-ui",
    ) -> Response:
        record = self.generated_image_records.get(token)
        if record is None:
            raise HTTPException(status_code=404, detail="Generated image token was not found.")

        file_path = self._generated_image_file_path(record)
        if file_path is not None:
            if download:
                return FileResponse(
                    file_path,
                    filename=sanitize_download_name(record.get("downloadName")),
                    content_disposition_type="attachment",
                )
            return FileResponse(file_path, headers={"Cache-Control": "no-store"})

        if not download:
            preview_path = self._generated_image_preview_path(record)
            if preview_path is not None:
                return FileResponse(preview_path, headers={"Cache-Control": "no-store"})

            browser_url = await self._resolve_generated_browser_url(record)
            if browser_url:
                return await self._fetch_image_proxy_response(
                    browser_url,
                    download_name=record.get("downloadName"),
                    as_download=False,
                    request_source=request_source,
                )

            return await self._fetch_image_proxy_response(
                record.get("previewUrl") or record.get("sourceUrl") or "",
                download_name=record.get("downloadName"),
                as_download=False,
                request_source=request_source,
            )

        try:
            existing_task = self.generated_image_tasks.get(token)
            if existing_task is not None and not existing_task.done():
                await existing_task
                record = self.generated_image_records.get(token, record)
            else:
                record = await self._cache_generated_image_record(token)
        except Exception as exc:
            self.generated_image_records.get(token, {})["error"] = str(exc)
            self.generated_image_records.get(token, {})["cacheStatus"] = "failed"
            raise HTTPException(
                status_code=409,
                detail=f"Generated image local save failed: {exc}",
            ) from exc

        file_path = self._generated_image_file_path(record)
        if file_path is None:
            raise HTTPException(
                status_code=409,
                detail=record.get("error") or "Generated image local save is still pending.",
            )

        return FileResponse(
            file_path,
            filename=sanitize_download_name(record.get("downloadName")),
            content_disposition_type="attachment",
        )

    async def ask(
        self,
        payload: ChatRequest,
        request_source: str = "local-ui",
    ) -> dict[str, object]:
        await self.ensure_ready()
        assert self.client is not None

        reauth_attempts = 0
        standard_image_fallback_attempted = False

        while True:
            retry_after_reauth = False
            retry_standard_image_path = False

            async with self._chat_lock:
                requested_image_use_pro = should_use_pro_image_path(
                    prompt=payload.prompt,
                    requested_model=payload.model,
                    explicit_use_pro=payload.use_pro,
                )
                prompt_looks_like_image = (
                    looks_like_image_prompt(payload.prompt) or requested_image_use_pro
                )
                image_use_pro = (
                    requested_image_use_pro and not standard_image_fallback_attempted
                )
                effective_model = self._resolve_effective_model(
                    payload.model,
                    use_pro=image_use_pro,
                )
                can_start_fresh_chat = hasattr(self.client, "start_chat")
                if (
                    payload.reset
                    or self.chat is None
                    or effective_model != self.current_model
                    or (prompt_looks_like_image and can_start_fresh_chat)
                ):
                    await self.reset(effective_model)
                    if prompt_looks_like_image:
                        append_runtime_log(
                            "INFO",
                            "Started a fresh Gemini chat session for image generation.",
                        )

                assert self.chat is not None
                request_tag = uuid4().hex[:10]
                request_files, request_dir = self._materialize_input_files(
                    payload.images,
                    request_tag,
                )
                previous_timeout = self.client.timeout
                previous_watchdog = self.client.watchdog_timeout
                image_mode_label = resolve_image_mode_label(
                    prompt_looks_like_image,
                    image_use_pro,
                )
                effective_timeout = max(payload.timeout_seconds, 60)
                self.client.timeout = effective_timeout
                self.client.watchdog_timeout = min(
                    self.client.timeout,
                    resolve_stream_watchdog_timeout(prompt_looks_like_image),
                )
                latest_output = None
                partial_text = ""
                partial_thoughts = ""
                retry_override = None if prompt_looks_like_image else 0

                try:
                    async def consume_stream() -> None:
                        nonlocal latest_output, partial_text, partial_thoughts
                        if prompt_looks_like_image:
                            append_runtime_log(
                                "INFO",
                                "Image prompt detected. "
                                f"Using {image_mode_label or 'Image generation'} mode.",
                            )
                        if request_files:
                            append_runtime_log(
                                "INFO",
                                f"Attached {len(request_files)} local input image(s) to the Gemini request.",
                            )
                        assert self.chat is not None
                        async with self._hold_session_activity(
                            f"chat request ({self.current_model})",
                            request_source,
                        ):
                            async for output in self.chat.send_message_stream(
                                payload.prompt,
                                files=request_files or None,
                                temporary=payload.temporary,
                                use_pro=image_use_pro,
                                timeout=self.client.timeout,
                                current_retry=retry_override,
                            ):
                                latest_output = output
                                partial_text += output.text_delta or ""
                                partial_thoughts += output.thoughts_delta or ""

                    await asyncio.wait_for(consume_stream(), timeout=effective_timeout + 15)
                except asyncio.TimeoutError as exc:
                    self.chat = self.client.start_chat(model=self.current_model)
                    raise TimeoutError(
                        "Gemini took too long to answer. Try gemini-3-flash, increase the timeout, "
                        "or refresh cookies if the session looks stale."
                    ) from exc
                except Exception as exc:
                    if should_retry_standard_image_generation(
                        prompt_looks_like_image=prompt_looks_like_image,
                        requested_use_pro=requested_image_use_pro,
                        active_use_pro=image_use_pro,
                        fallback_attempted=standard_image_fallback_attempted,
                        output=latest_output,
                    ):
                        retry_standard_image_path = True
                        append_runtime_log(
                            "WARNING",
                            "Nano Banana Pro path returned text without images. Retrying once with the standard Gemini image-generation path.",
                        )
                    elif latest_output is not None or partial_text or partial_thoughts:
                        append_runtime_log(
                            "WARNING",
                            f"Gemini sync request ended after partial output: {exc}",
                        )
                        images = await self._materialize_images(
                            list(getattr(latest_output, "images", None) or []),
                            request_tag,
                            defer_generated_cache=True,
                            request_source=request_source,
                            use_pro=image_use_pro,
                        )
                        metadata = list(getattr(latest_output, "metadata", None) or [])
                        history = await self.check_history(
                            metadata[0] if metadata else "",
                            payload.temporary,
                            wait_seconds=0,
                            request_source=request_source,
                        )
                        warning = (
                            "Gemini stopped after returning a partial response."
                            if not prompt_looks_like_image
                            else (
                                "Gemini returned text, but no usable image payload. "
                                "This usually means the current session does not have image generation access."
                            )
                        )
                        return {
                            "text": getattr(latest_output, "text", None) or partial_text,
                            "thoughts": getattr(latest_output, "thoughts", None) or partial_thoughts or None,
                            "metadata": metadata,
                            "model": self.current_model,
                            "images": images,
                            "history": history,
                            "partial": True,
                            "warning": warning,
                            "imageMode": image_mode_label,
                        }

                    if (
                        reauth_attempts < MAX_AUTO_REAUTH_RETRIES
                        and self._should_auto_reprobe_stream_error(
                            exc,
                            latest_output=latest_output,
                            partial_text=partial_text,
                            partial_thoughts=partial_thoughts,
                        )
                    ):
                        retry_after_reauth = True
                    else:
                        raise
                finally:
                    self.client.timeout = previous_timeout
                    self.client.watchdog_timeout = previous_watchdog
                    self._cleanup_input_files(request_dir)

                if retry_after_reauth:
                    pass
                else:
                    if latest_output is None:
                        raise RuntimeError("Gemini returned no output for this request.")

                image_warning = None
                if retry_after_reauth:
                    pass
                else:
                    if should_retry_standard_image_generation(
                        prompt_looks_like_image=prompt_looks_like_image,
                        requested_use_pro=requested_image_use_pro,
                        active_use_pro=image_use_pro,
                        fallback_attempted=standard_image_fallback_attempted,
                        output=latest_output,
                    ):
                        retry_standard_image_path = True
                        append_runtime_log(
                            "WARNING",
                            "Nano Banana Pro path returned text without images. Retrying once with the standard Gemini image-generation path.",
                        )
                    elif prompt_looks_like_image and not output_has_images(latest_output):
                        image_warning = (
                            "Gemini returned text without image payloads. Nano Banana Pro appears unavailable through the Gemini web reverse API for this session right now."
                            if image_use_pro
                            else "Gemini returned text without image payloads. The current session may not have image generation access."
                        )
                        append_runtime_log(
                            "WARNING",
                            "Gemini finished the image prompt without image payloads.",
                        )

                    if not retry_standard_image_path:
                        images = await self._materialize_images(
                            latest_output.images,
                            request_tag,
                            defer_generated_cache=True,
                            request_source=request_source,
                            use_pro=image_use_pro,
                        )
                        append_runtime_log(
                            "INFO",
                            f"Gemini completed the request with {len(images)} image(s) and "
                            f"{'text output' if (latest_output.text or '').strip() else 'no text body'}.",
                        )
                        history = await self.check_history(
                            latest_output.metadata[0] if latest_output.metadata else "",
                            payload.temporary,
                            wait_seconds=0,
                            request_source=request_source,
                        )
                        return {
                            "text": latest_output.text,
                            "thoughts": latest_output.thoughts,
                            "metadata": latest_output.metadata,
                            "model": self.current_model,
                            "imageMode": image_mode_label,
                            "images": images,
                            "history": history,
                            "warning": image_warning,
                        }

            if retry_standard_image_path:
                standard_image_fallback_attempted = True
                continue
            if retry_after_reauth and reauth_attempts < MAX_AUTO_REAUTH_RETRIES:
                append_runtime_log("WARNING", AUTO_REAUTH_STATUS_MESSAGE)
                await self.reprobe_auth(force=True)
                reauth_attempts += 1
                continue

    async def _materialize_images(
        self,
        images: list[Any],
        request_tag: str,
        defer_generated_cache: bool = False,
        request_source: str = "local-ui",
        use_pro: bool = False,
    ) -> list[dict[str, Any]]:
        if not images:
            return []

        target_dir = IMAGE_CACHE_DIR / request_tag
        target_dir.mkdir(parents=True, exist_ok=True)

        async def _save_single(index: int, image: Any) -> dict[str, Any]:
            if isinstance(image, GeneratedImage):
                image_mode = resolve_image_mode_label(True, use_pro)
                record = self._register_generated_image(
                    image,
                    request_tag,
                    index,
                    request_source=request_source,
                    use_pro=use_pro,
                    image_mode=image_mode,
                )
                if defer_generated_cache:
                    browser_url = await self._resolve_generated_browser_url(record)
                    if not browser_url:
                        try:
                            await asyncio.wait_for(
                                self._cache_generated_image_preview_record(record["token"]),
                                timeout=6,
                            )
                            append_runtime_log(
                                "INFO",
                                f"Rendered Gemini image {index + 1} locally.",
                            )
                        except Exception as exc:
                            append_runtime_log(
                                "DEBUG",
                                "Local preview cache for Gemini generated image "
                                f"{index + 1} is still pending. {exc}",
                            )
                    self._start_generated_image_cache(
                        record["token"],
                        delay_seconds=1.5 if browser_url else 0.0,
                    )
                    if browser_url or self._generated_image_preview_path(record) is not None:
                        append_runtime_log(
                            "INFO",
                            "Preview is ready for Gemini generated image "
                            f"{index + 1}. The local download copy is saving in the background.",
                        )
                    else:
                        append_runtime_log(
                            "WARNING",
                            "Gemini generated image "
                            f"{index + 1} did not expose a direct preview URL yet. The local save will keep trying in the background.",
                        )
                    return self._build_generated_image_payload(record)
                try:
                    await self._cache_generated_image_record(record["token"])
                except Exception as exc:
                    await self._resolve_generated_browser_url(record)
                    record["error"] = str(exc)
                    record["cacheStatus"] = "failed"
                    append_runtime_log(
                        "WARNING",
                        "Could not cache Gemini generated image "
                        f"{index + 1} immediately. The app will retry through the authenticated image endpoint. {exc}",
                    )
                return self._build_generated_image_payload(record)

            try:
                shared_http_client = self.client.client if self.client and self.client.client else None
                saved_path = await image.save(
                    path=str(target_dir),
                    filename=f"image_{index}",
                    verbose=False,
                    client=shared_http_client,
                )
                relative_path = Path(saved_path).relative_to(IMAGE_CACHE_DIR).as_posix()
                quality = normalize_image_quality(getattr(image, "saved_quality", None))
                append_runtime_log(
                    "INFO",
                    f"Saved Gemini image {index + 1} to local cache ({quality}).",
                )
                download_name = Path(saved_path).name
                return {
                    "title": image.title,
                    "alt": image.alt,
                    "sourceUrl": image.url,
                    "proxyUrl": f"/generated/{relative_path}",
                    "downloadUrl": f"/generated/{relative_path}",
                    "downloadName": download_name,
                    "quality": quality,
                    "cached": True,
                    "downloadReady": True,
                    "cacheStatus": "ready",
                    "serverManaged": False,
                }
            except Exception as exc:
                append_runtime_log(
                    "WARNING",
                    f"Failed to cache Gemini image {index + 1}; falling back to proxy URL. {exc}",
                )
                fallback_name = sanitize_download_name(
                    f"{Path(image.alt or image.title or f'image_{index + 1}').stem}.png"
                )
                return {
                    "title": image.title,
                    "alt": image.alt,
                    "sourceUrl": image.url,
                    "proxyUrl": f"/api/image?url={quote(image.url, safe='')}",
                    "downloadUrl": (
                        f"/api/image?url={quote(image.url, safe='')}"
                        f"&download=true&filename={quote(fallback_name, safe='')}"
                    ),
                    "downloadName": fallback_name,
                    "quality": "source",
                    "cached": False,
                    "downloadReady": True,
                    "cacheStatus": "ready",
                    "serverManaged": False,
                    "error": str(exc),
                }

        return await asyncio.gather(*(_save_single(i, image) for i, image in enumerate(images)))

    async def check_history(
        self,
        cid: str,
        temporary: bool,
        wait_seconds: float = 0,
        request_source: str = "local-ui",
    ) -> dict[str, Any]:
        if temporary:
            return build_history_state(
                cid=cid,
                status="skipped",
                message="Temporary mode is on, so Gemini history is intentionally skipped.",
                checked=True,
                saved=False,
            )

        if not cid:
            return build_history_state(
                cid="",
                status="missing",
                message="Gemini did not return a chat ID, so this reply cannot be checked in history.",
                checked=True,
                saved=False,
            )

        if self.account_status == AccountStatus.UNAUTHENTICATED:
            return build_history_state(
                cid=cid,
                status="blocked",
                message="This cookie session can answer prompts, but Gemini reports it as unauthenticated for history sync.",
                checked=True,
                saved=False,
            )

        assert self.client is not None
        deadline = time.monotonic() + max(0, wait_seconds)
        attempts = 0

        while True:
            attempts += 1
            try:
                async with self._hold_session_activity(
                    f"history check ({cid or 'missing-cid'})",
                    request_source,
                ):
                    history = await self.client.read_chat(cid, limit=4)
                    if history and history.turns:
                        await self.client._fetch_recent_chats(20)
                        listed = any(chat.cid == cid for chat in (self.client.list_chats() or []))
                        append_runtime_log("INFO", f"Verified Gemini history for chat {cid}.")
                        result = build_history_state(
                            cid=cid,
                            status="saved",
                            message="Conversation was verified in Gemini history.",
                            checked=True,
                            saved=True,
                        )
                        result["listed"] = listed
                        return result
            except Exception as exc:
                append_runtime_log("DEBUG", f"History verification retry for {cid}: {exc}")

            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(2)

        pending_message = "Gemini returned a chat ID. History confirmation is still pending."
        if attempts > 1:
            append_runtime_log("WARNING", f"Could not verify Gemini history for chat {cid} yet.")
            pending_message = "Gemini returned a chat ID, but history could not be confirmed yet."

        return build_history_state(
            cid=cid,
            status="pending",
            message=pending_message,
            checked=False,
            saved=False,
        )

    async def stream_chat_events(
        self,
        payload: ChatRequest,
        request_source: str = "local-ui",
    ) -> AsyncGenerator[dict[str, Any], None]:
        await self.ensure_ready()
        assert self.client is not None
        reauth_attempts = 0
        standard_image_fallback_attempted = False

        while True:
            retry_after_reauth = False
            retry_standard_image_path = False
            log_cursor = get_latest_log_id()

            async with self._chat_lock:
                queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
                request_started = time.monotonic()
                request_tag = uuid4().hex[:10]
                request_files, request_dir = self._materialize_input_files(
                    payload.images,
                    request_tag,
                )
                previous_timeout = self.client.timeout
                previous_watchdog = self.client.watchdog_timeout
                partial_text = ""
                partial_thoughts = ""
                seen_image_notice = False
                preview_images: list[dict[str, Any]] = []
                requested_image_use_pro = should_use_pro_image_path(
                    prompt=payload.prompt,
                    requested_model=payload.model,
                    explicit_use_pro=payload.use_pro,
                )
                prompt_looks_like_image = (
                    looks_like_image_prompt(payload.prompt) or requested_image_use_pro
                )
                image_use_pro = (
                    requested_image_use_pro and not standard_image_fallback_attempted
                )
                effective_model = self._resolve_effective_model(
                    payload.model,
                    use_pro=image_use_pro,
                )
                image_mode_label = resolve_image_mode_label(
                    prompt_looks_like_image,
                    image_use_pro,
                )
                retry_override = None if prompt_looks_like_image else 0

                async def worker() -> None:
                    nonlocal partial_text
                    nonlocal partial_thoughts
                    nonlocal seen_image_notice
                    nonlocal preview_images
                    nonlocal retry_after_reauth
                    nonlocal retry_standard_image_path
                    latest_output = None

                    async def emit_final(
                        *,
                        output: Any | None,
                        partial: bool = False,
                        warning: str | None = None,
                    ) -> None:
                        metadata = list(getattr(output, "metadata", None) or [])
                        cid = metadata[0] if metadata else ""
                        images = preview_images or await self._materialize_images(
                            list(getattr(output, "images", None) or []),
                            request_tag,
                            defer_generated_cache=True,
                            request_source=request_source,
                            use_pro=image_use_pro,
                        )
                        history = await self.check_history(
                            cid,
                            payload.temporary,
                            wait_seconds=0,
                            request_source=request_source,
                        )

                        await queue.put(
                            {
                                "type": "final",
                                "text": getattr(output, "text", None) or partial_text,
                                "thoughts": getattr(output, "thoughts", None) or partial_thoughts or None,
                                "images": images,
                                "metadata": metadata,
                                "model": self.current_model,
                                "imageMode": image_mode_label,
                                "history": history,
                                "partial": partial,
                                "warning": warning,
                            }
                        )

                    try:
                        can_start_fresh_chat = hasattr(self.client, "start_chat")
                        if (
                            payload.reset
                            or self.chat is None
                            or effective_model != self.current_model
                            or (prompt_looks_like_image and can_start_fresh_chat)
                        ):
                            await self.reset(effective_model)
                            session_message = (
                                f"Started a fresh image chat session for model {self.current_model}."
                                if prompt_looks_like_image
                                else f"Started a fresh chat session for model {self.current_model}."
                            )
                            if prompt_looks_like_image:
                                append_runtime_log(
                                    "INFO",
                                    "Started a fresh Gemini chat session for image generation.",
                                )
                            await queue.put(
                                {
                                    "type": "status",
                                    "message": session_message,
                                }
                            )

                        if payload.temporary:
                            await queue.put(
                                {
                                    "type": "status",
                                    "message": "Temporary mode is on. This reply will not be saved to Gemini history.",
                                }
                            )
                        else:
                            await queue.put(
                                {
                                    "type": "status",
                                    "message": "Temporary mode is off. I will return the reply immediately, then keep checking whether Gemini history confirms it.",
                                }
                            )

                        if self.account_status and self.account_status != AccountStatus.AVAILABLE:
                            await queue.put(
                                {
                                    "type": "status",
                                    "level": "WARNING",
                                    "message": (
                                        f"Gemini account status is {self.account_status.name}. "
                                        "Replies may still work, but history sync and media features can fail."
                                    ),
                                }
                            )

                        if prompt_looks_like_image and "thinking" in self.current_model:
                            await queue.put(
                                {
                                    "type": "status",
                                    "level": "WARNING",
                                    "message": "Thinking model selected for an image-style prompt. Gemini Flash or Pro is usually more reliable for image generation.",
                                }
                            )
                        if prompt_looks_like_image:
                            await queue.put(
                                {
                                    "type": "status",
                                    "message": (
                                        f"{image_mode_label or 'Image generation'} can take several minutes and may require "
                                        "internal retries before Gemini commits the final image response."
                                    ),
                                }
                            )

                        self.client.timeout = max(payload.timeout_seconds, 60)
                        self.client.watchdog_timeout = min(
                            self.client.timeout,
                            resolve_stream_watchdog_timeout(prompt_looks_like_image),
                        )
                        append_runtime_log(
                            "INFO",
                            f"Sending prompt to Gemini with model {self.current_model} (temporary={payload.temporary}).",
                        )
                        await queue.put(
                            {
                                "type": "status",
                                "message": (
                                    f"Prompt is being sent to Gemini with model {self.current_model}."
                                ),
                            }
                        )
                        if request_files:
                            append_runtime_log(
                                "INFO",
                                f"Attached {len(request_files)} local input image(s) to the Gemini request.",
                            )
                        if prompt_looks_like_image:
                            append_runtime_log(
                                "INFO",
                                "Image prompt detected. "
                                f"Using {image_mode_label or 'Image generation'} mode.",
                            )

                        assert self.chat is not None
                        async with self._hold_session_activity(
                            f"chat stream ({self.current_model})",
                            request_source,
                        ):
                            async for output in self.chat.send_message_stream(
                                payload.prompt,
                                files=request_files or None,
                                temporary=payload.temporary,
                                use_pro=image_use_pro,
                                timeout=self.client.timeout,
                                current_retry=retry_override,
                            ):
                                latest_output = output

                                if output.thoughts_delta:
                                    partial_thoughts += output.thoughts_delta
                                    await queue.put(
                                        {
                                            "type": "delta",
                                            "thoughts_delta": output.thoughts_delta,
                                            "message": "Gemini sent a thoughts chunk.",
                                        }
                                    )

                                if output.text_delta:
                                    partial_text += output.text_delta
                                    if not (
                                        prompt_looks_like_image
                                        and image_use_pro
                                        and not seen_image_notice
                                    ):
                                        await queue.put(
                                            {
                                                "type": "delta",
                                                "text_delta": output.text_delta,
                                                "message": "Gemini sent a text chunk.",
                                            }
                                        )

                                if output.images and not seen_image_notice:
                                    seen_image_notice = True
                                    preview_images = await self._materialize_images(
                                        list(output.images),
                                        request_tag,
                                        defer_generated_cache=True,
                                        request_source=request_source,
                                        use_pro=image_use_pro,
                                    )
                                    await queue.put(
                                        {
                                            "type": "images_preview",
                                            "images": preview_images,
                                            "imageMode": image_mode_label,
                                            "message": (
                                                f"Gemini attached {len(output.images)} image(s). "
                                                "Showing the preview now while the local file saves in the background."
                                            ),
                                        }
                                    )

                        if latest_output is None:
                            raise RuntimeError("Gemini returned no output for this request.")

                        if should_retry_standard_image_generation(
                            prompt_looks_like_image=prompt_looks_like_image,
                            requested_use_pro=requested_image_use_pro,
                            active_use_pro=image_use_pro,
                            fallback_attempted=standard_image_fallback_attempted,
                            output=latest_output,
                        ):
                            retry_standard_image_path = True
                            retry_message = (
                                "Nano Banana Pro returned text without images. "
                                "Retrying once with the standard Gemini image-generation path."
                            )
                            append_runtime_log("WARNING", retry_message)
                            await queue.put(
                                {
                                    "type": "retry",
                                    "level": "WARNING",
                                    "message": retry_message,
                                    "imageMode": "Image generation",
                                }
                            )
                            return

                        image_warning = None
                        if prompt_looks_like_image and not output_has_images(latest_output):
                            image_warning = (
                                "Gemini returned text without image payloads. Nano Banana Pro appears unavailable through the Gemini web reverse API for this session right now."
                                if image_use_pro
                                else "Gemini returned text without image payloads. The current session may not have image generation access."
                            )
                            await queue.put(
                                {
                                    "type": "status",
                                    "level": "WARNING",
                                    "message": image_warning,
                                }
                            )

                        append_runtime_log(
                            "INFO",
                            f"Gemini completed the request with {len(getattr(latest_output, 'images', None) or [])} image(s) and "
                            f"{'text output' if (getattr(latest_output, 'text', '') or '').strip() else 'no text body'}.",
                        )
                        await emit_final(output=latest_output, warning=image_warning)
                    except Exception as exc:
                        error_text = str(exc)
                        if should_retry_standard_image_generation(
                            prompt_looks_like_image=prompt_looks_like_image,
                            requested_use_pro=requested_image_use_pro,
                            active_use_pro=image_use_pro,
                            fallback_attempted=standard_image_fallback_attempted,
                            output=latest_output,
                        ):
                            retry_standard_image_path = True
                            retry_message = (
                                "Nano Banana Pro returned text without images. "
                                "Retrying once with the standard Gemini image-generation path."
                            )
                            append_runtime_log("WARNING", retry_message)
                            await queue.put(
                                {
                                    "type": "retry",
                                    "level": "WARNING",
                                    "message": retry_message,
                                    "imageMode": "Image generation",
                                }
                            )
                        elif latest_output is not None or partial_text or partial_thoughts:
                            warning = (
                                "Gemini stopped after returning a partial response."
                                if not prompt_looks_like_image
                                else (
                                    "Gemini returned text, but no usable image payload. "
                                    "This usually means the current session does not have image generation access."
                                )
                            )
                            append_runtime_log(
                                "WARNING",
                                f"Gemini stream ended after partial output: {error_text}",
                            )
                            await queue.put(
                                {
                                    "type": "status",
                                    "level": "WARNING",
                                    "message": warning,
                                }
                            )
                            await emit_final(
                                output=latest_output,
                                partial=True,
                                warning=warning,
                            )
                        elif (
                            reauth_attempts < MAX_AUTO_REAUTH_RETRIES
                            and self._should_auto_reprobe_stream_error(
                                exc,
                                latest_output=latest_output,
                                partial_text=partial_text,
                                partial_thoughts=partial_thoughts,
                            )
                        ):
                            retry_after_reauth = True
                            append_runtime_log("WARNING", AUTO_REAUTH_STATUS_MESSAGE)
                            await queue.put(
                                {
                                    "type": "status",
                                    "level": "WARNING",
                                    "message": AUTO_REAUTH_STATUS_MESSAGE,
                                }
                            )
                        else:
                            append_runtime_log("ERROR", f"Gemini request failed: {error_text}")
                            await queue.put(
                                {
                                    "type": "error",
                                    "message": error_text,
                                }
                            )
                    finally:
                        self.client.timeout = previous_timeout
                        self.client.watchdog_timeout = previous_watchdog
                        await queue.put({"type": "done"})

                worker_task = asyncio.create_task(worker())

                try:
                    while True:
                        new_logs, log_cursor = consume_logs(log_cursor)
                        for entry in new_logs:
                            yield {
                                "type": "log",
                                "id": entry["id"],
                                "level": entry["level"],
                                "message": entry["message"],
                                "ts": entry["ts"],
                            }

                        try:
                            event = await asyncio.wait_for(queue.get(), timeout=3)
                        except asyncio.TimeoutError:
                            elapsed = int(time.monotonic() - request_started)
                            waiting_message = (
                                f"Still waiting for Gemini after {elapsed}s. No visible output chunk yet."
                                if not partial_text and not partial_thoughts
                                else f"Gemini is still working after {elapsed}s."
                            )
                            yield {
                                "type": "heartbeat",
                                "message": waiting_message,
                                "elapsed": elapsed,
                            }
                            continue

                        if event["type"] == "done":
                            trailing_logs, log_cursor = consume_logs(log_cursor)
                            for entry in trailing_logs:
                                yield {
                                    "type": "log",
                                    "id": entry["id"],
                                    "level": entry["level"],
                                    "message": entry["message"],
                                    "ts": entry["ts"],
                                }
                            break

                        yield event
                finally:
                    await worker_task
                    self._cleanup_input_files(request_dir)

            if retry_after_reauth and reauth_attempts < MAX_AUTO_REAUTH_RETRIES:
                await self.reprobe_auth(force=True)
                reauth_attempts += 1
                yield {
                    "type": "status",
                    "message": "Re-auth completed. Retrying the same Gemini request once.",
                }
                continue
            if retry_standard_image_path:
                standard_image_fallback_attempted = True
                yield {
                    "type": "status",
                    "level": "WARNING",
                    "message": "Retrying with standard Gemini image generation.",
                }
                continue
            break


service = GeminiLocalService()
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
INPUT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Gemini Local UI", version=APP_VERSION)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/generated", StaticFiles(directory=IMAGE_CACHE_DIR), name="generated")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await service.close()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, max-age=0, must-revalidate"},
    )


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/status")
async def api_status() -> dict[str, object]:
    return await service.status()


@app.get("/api/logs")
async def api_logs(
    after_id: int = Query(default=0, ge=0),
    limit: int = Query(default=80, ge=1, le=200),
) -> dict[str, object]:
    logs, latest_id = consume_logs(after_id)
    if len(logs) > limit:
        logs = logs[-limit:]
    return {
        "logs": logs,
        "latestId": latest_id,
    }


@app.get("/api/history/{cid}")
async def api_history(request: Request, cid: str, temporary: bool = False) -> dict[str, object]:
    try:
        await service.ensure_ready()
        return await service.check_history(
            cid,
            temporary=temporary,
            wait_seconds=0,
            request_source=infer_request_source(request),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/cookies")
async def api_cookies(payload: CookieUpdateRequest) -> dict[str, object]:
    try:
        return await service.update_credentials(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/reprobe-auth")
async def api_reprobe_auth() -> dict[str, object]:
    try:
        reprobed = await service.reprobe_auth(force=True)
        logs, _ = consume_logs(max(get_latest_log_id() - 12, 0))
        payload = service._build_status_payload(
            ready=service.client is not None,
            error=service.boot_error,
            recent_logs=logs,
            reprobed=reprobed,
        )
        payload["ok"] = True
        return payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/reset")
async def api_reset(payload: ResetRequest) -> dict[str, object]:
    try:
        return await service.reset(payload.model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/chat")
async def api_chat(request: Request, payload: ChatRequest) -> dict[str, object]:
    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        return await service.ask(payload, request_source=infer_request_source(request))
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/chat/stream")
async def api_chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    request_source = infer_request_source(request)

    async def event_source() -> AsyncGenerator[bytes, None]:
        append_runtime_log(
            "INFO",
            f"Local stream endpoint accepted a prompt from {request_source}.",
        )
        yield (
            json.dumps(
                {
                    "type": "status",
                    "message": "Local backend received the request and is opening the Gemini stream.",
                },
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        async for event in service.stream_chat_events(payload, request_source=request_source):
            yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(
        event_source(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/api/generated-image/{token}")
async def api_generated_image(request: Request, token: str, download: bool = False) -> Response:
    return await service.get_generated_image_response(
        token,
        download=download,
        request_source=infer_request_source(request),
    )


@app.get("/api/generated-image-status/{token}")
async def api_generated_image_status(token: str) -> dict[str, object]:
    return await service.get_generated_image_status(token)


@app.get("/api/image")
async def api_image(
    request: Request,
    url: str,
    download: bool = False,
    filename: str | None = Query(default=None),
) -> Response:
    try:
        return await service._fetch_image_proxy_response(
            url,
            download_name=filename,
            as_download=download,
            request_source=infer_request_source(request),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Image proxy failed: {exc}") from exc

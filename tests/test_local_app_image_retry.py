import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import local_app  # noqa: E402


class FakeOutput:
    def __init__(self, text="done", images=None, metadata=None):
        self.text = text
        self.text_delta = text
        self.thoughts = ""
        self.thoughts_delta = ""
        self.images = list(images or [])
        self.metadata = list(metadata or [])


class FakeChat:
    def __init__(self, output=None):
        self.calls = []
        self.output = output or FakeOutput()

    async def send_message_stream(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        yield self.output


class FakeAbortChat:
    def __init__(self):
        self.calls = []

    async def send_message_stream(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        local_app.append_runtime_log(
            "DEBUG",
            "Stream suspended (completed=False, final_chunk=False, thinking=False, queueing=False). "
            "No CID found to recover. (Request ID: 123)",
        )
        if False:
            yield None
        raise RuntimeError("The original request may have been silently aborted by Google.")


class FakeSavedImage:
    def __init__(self, saved_path: str):
        self.saved_path = saved_path
        self.title = "[Generated Image 0]"
        self.alt = "red-lantern"
        self.url = "https://lh3.googleusercontent.com/gg-dl/token"
        self.saved_quality = "full"

    async def save(self, **kwargs):
        return self.saved_path


class FakeGeneratedImage(local_app.GeneratedImage):
    async def save(
        self,
        path="temp",
        filename=None,
        verbose=False,
        client=None,
        **kwargs,
    ):
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        dest = path_obj / f"{Path(filename).stem}.png"
        dest.write_bytes(b"png")
        self.saved_quality = "fhd"
        self.preview_url = self.preview_url or self.url
        return str(dest.resolve())


class FakeProxyResponse:
    def __init__(self, status_code: int, content_type: str, text: str = "", content: bytes = b"", reason: str = "OK"):
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.text = text
        self.content = content
        self.reason = reason


class FakeProxyClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []

    async def get(self, url, headers=None, **kwargs):
        self.urls.append(url)
        if not self.responses:
            raise AssertionError("No more fake responses configured.")
        return self.responses.pop(0)

    async def close(self):
        return None


def make_fake_client(http_client=None):
    return SimpleNamespace(proxy=None, client=http_client, cookies={})


class TestLocalAppImageRetry(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        local_app.LOG_BUFFER.clear()

    def test_nano_banana_pro_model_prefers_advanced_pro_entry(self):
        model = local_app.resolve_nano_banana_pro_model(
            [
                {"name": "gemini-3-flash", "label": "Nhanh"},
                {"name": "gemini-3-pro-plus", "label": "Pro"},
                {"name": "gemini-3-pro-advanced", "label": "Pro"},
            ]
        )

        self.assertEqual(model, "gemini-3-pro-advanced")

    def test_nano_banana_pro_model_accepts_3_1_pro_name(self):
        model = local_app.resolve_nano_banana_pro_model(
            [
                {"name": "gemini-3.1-pro", "label": "Pro"},
                {"name": "gemini-3-flash", "label": "Nhanh"},
            ]
        )

        self.assertEqual(model, "gemini-3.1-pro")

    def test_nano_banana_pro_model_returns_none_when_registry_has_no_pro(self):
        model = local_app.resolve_nano_banana_pro_model(
            [
                {"name": "gemini-3-flash", "label": "Nhanh"},
                {"name": "gemini-3-flash-thinking", "label": "Tư duy"},
            ]
        )

        self.assertIsNone(model)

    def test_image_prompt_with_selected_pro_model_auto_uses_pro_image_path(self):
        self.assertTrue(
            local_app.should_use_pro_image_path(
                prompt="Generate an image of a red paper lantern.",
                requested_model="gemini-3-pro-advanced",
                explicit_use_pro=False,
            )
        )

    def test_stream_watchdog_uses_shorter_suspended_detection_window(self):
        self.assertEqual(local_app.resolve_stream_watchdog_timeout(False), 20.0)
        self.assertEqual(local_app.resolve_stream_watchdog_timeout(True), 45.0)

    async def test_image_prompt_uses_default_image_path_and_honors_timeout(self):
        service = local_app.GeminiLocalService()
        fake_chat = FakeChat(FakeOutput(images=[object()]))
        fake_client = SimpleNamespace(timeout=30.0, watchdog_timeout=30.0)

        service.client = fake_client
        service.chat = fake_chat
        service.current_model = "gemini-3-flash"
        service.ensure_ready = AsyncMock()
        service._materialize_images = AsyncMock(return_value=[{"token": "img"}])
        service.check_history = AsyncMock(
            return_value=local_app.build_history_state(
                cid="",
                status="missing",
                message="missing",
                checked=True,
                saved=False,
            )
        )

        payload = local_app.ChatRequest(
            prompt="Generate an image of a red paper lantern.",
            model="gemini-3-flash",
            timeout_seconds=150,
        )

        result = await service.ask(payload)

        self.assertEqual(result["text"], "done")
        self.assertEqual(len(fake_chat.calls), 1)
        _, kwargs = fake_chat.calls[0]
        sent_prompt = fake_chat.calls[0][0]
        self.assertEqual(sent_prompt, "Generate an image of a red paper lantern.")
        self.assertIsNone(kwargs["current_retry"])
        self.assertFalse(kwargs["use_pro"])
        self.assertEqual(kwargs["timeout"], 150)
        self.assertEqual(fake_client.timeout, 30.0)
        self.assertEqual(fake_client.watchdog_timeout, 30.0)

    async def test_image_prompt_passes_use_pro_when_enabled(self):
        service = local_app.GeminiLocalService()
        fake_chat = FakeChat(FakeOutput(images=[object()]))
        fake_client = SimpleNamespace(
            timeout=30.0,
            watchdog_timeout=30.0,
            start_chat=lambda model: fake_chat,
        )

        service.client = fake_client
        service.chat = fake_chat
        service.current_model = "gemini-3-flash"
        service.available_models = [
            {"name": "gemini-3-pro-advanced", "label": "Pro"},
            {"name": "gemini-3-flash", "label": "Nhanh"},
        ]
        service.ensure_ready = AsyncMock()
        service._materialize_images = AsyncMock(return_value=[{"token": "img"}])
        service.check_history = AsyncMock(
            return_value=local_app.build_history_state(
                cid="",
                status="missing",
                message="missing",
                checked=True,
                saved=False,
            )
        )

        payload = local_app.ChatRequest(
            prompt="Generate an image of a red paper lantern.",
            model="gemini-3-flash",
            timeout_seconds=150,
            use_pro=True,
        )

        result = await service.ask(payload)

        self.assertEqual(result["imageMode"], "Image generation Pro")
        sent_prompt = fake_chat.calls[0][0]
        self.assertEqual(sent_prompt, "Generate an image of a red paper lantern.")
        _, kwargs = fake_chat.calls[0]
        self.assertTrue(kwargs["use_pro"])
        self.assertEqual(service.current_model, "gemini-3-pro-advanced")

    async def test_image_prompt_auto_enables_use_pro_for_selected_pro_model(self):
        service = local_app.GeminiLocalService()
        fake_chat = FakeChat(FakeOutput(images=[object()]))
        fake_client = SimpleNamespace(
            timeout=30.0,
            watchdog_timeout=30.0,
            start_chat=lambda model: fake_chat,
        )

        service.client = fake_client
        service.chat = fake_chat
        service.current_model = "gemini-3-flash"
        service.available_models = [
            {"name": "gemini-3-pro-advanced", "label": "Pro"},
            {"name": "gemini-3-flash", "label": "Nhanh"},
        ]
        service.ensure_ready = AsyncMock()
        service._materialize_images = AsyncMock(return_value=[{"token": "img"}])
        service.check_history = AsyncMock(
            return_value=local_app.build_history_state(
                cid="",
                status="missing",
                message="missing",
                checked=True,
                saved=False,
            )
        )

        payload = local_app.ChatRequest(
            prompt="Generate an image of a red paper lantern.",
            model="gemini-3-pro-advanced",
            timeout_seconds=150,
        )

        result = await service.ask(payload)

        self.assertEqual(result["imageMode"], "Image generation Pro")
        _, kwargs = fake_chat.calls[0]
        self.assertTrue(kwargs["use_pro"])
        self.assertEqual(service.current_model, "gemini-3-pro-advanced")

    async def test_use_pro_checkbox_forces_pro_image_path_even_when_prompt_detector_misses(self):
        service = local_app.GeminiLocalService()
        fake_chat = FakeChat(FakeOutput(images=[object()]))
        fake_client = SimpleNamespace(
            timeout=30.0,
            watchdog_timeout=30.0,
            start_chat=lambda model: fake_chat,
        )

        service.client = fake_client
        service.chat = fake_chat
        service.current_model = "gemini-3-flash"
        service.available_models = [
            {"name": "gemini-3-pro-advanced", "label": "Pro"},
            {"name": "gemini-3-flash", "label": "Nhanh"},
        ]
        service.ensure_ready = AsyncMock()
        service._materialize_images = AsyncMock(return_value=[{"token": "img"}])
        service.check_history = AsyncMock(
            return_value=local_app.build_history_state(
                cid="",
                status="missing",
                message="missing",
                checked=True,
                saved=False,
            )
        )

        payload = local_app.ChatRequest(
            prompt="Make this more dramatic.",
            model="gemini-3-flash",
            timeout_seconds=150,
            use_pro=True,
        )

        result = await service.ask(payload)

        self.assertEqual(result["imageMode"], "Image generation Pro")
        sent_prompt = fake_chat.calls[0][0]
        self.assertEqual(sent_prompt, "Make this more dramatic.")
        _, kwargs = fake_chat.calls[0]
        self.assertTrue(kwargs["use_pro"])
        self.assertIsNone(kwargs["current_retry"])
        self.assertEqual(service.current_model, "gemini-3-pro-advanced")

    async def test_stream_use_pro_checkbox_forces_pro_image_path_even_when_prompt_detector_misses(self):
        service = local_app.GeminiLocalService()
        fake_chat = FakeChat(FakeOutput(images=[object()]))
        fake_client = SimpleNamespace(
            timeout=30.0,
            watchdog_timeout=30.0,
            start_chat=lambda model: fake_chat,
        )

        service.client = fake_client
        service.chat = fake_chat
        service.current_model = "gemini-3-flash"
        service.available_models = [
            {"name": "gemini-3-pro-advanced", "label": "Pro"},
            {"name": "gemini-3-flash", "label": "Nhanh"},
        ]
        service.ensure_ready = AsyncMock()
        service._materialize_images = AsyncMock(return_value=[{"token": "img"}])
        service.check_history = AsyncMock(
            return_value=local_app.build_history_state(
                cid="",
                status="missing",
                message="missing",
                checked=True,
                saved=False,
            )
        )

        payload = local_app.ChatRequest(
            prompt="Make this more dramatic.",
            model="gemini-3-flash",
            timeout_seconds=150,
            use_pro=True,
        )

        events = []
        async for event in service.stream_chat_events(payload):
            events.append(event)

        self.assertEqual(len(fake_chat.calls), 1)
        sent_prompt = fake_chat.calls[0][0]
        self.assertEqual(sent_prompt, "Make this more dramatic.")
        self.assertTrue(fake_chat.calls[0][1]["use_pro"])
        self.assertIsNone(fake_chat.calls[0][1]["current_retry"])
        self.assertEqual(service.current_model, "gemini-3-pro-advanced")
        final_event = next(event for event in events if event.get("type") == "final")
        self.assertEqual(final_event["imageMode"], "Image generation Pro")

    async def test_stream_image_prompt_auto_enables_use_pro_for_selected_pro_model(self):
        service = local_app.GeminiLocalService()
        fake_chat = FakeChat(FakeOutput(images=[object()]))
        fake_client = SimpleNamespace(
            timeout=30.0,
            watchdog_timeout=30.0,
            start_chat=lambda model: fake_chat,
        )

        service.client = fake_client
        service.chat = fake_chat
        service.current_model = "gemini-3-flash"
        service.available_models = [
            {"name": "gemini-3-pro-advanced", "label": "Pro"},
            {"name": "gemini-3-flash", "label": "Nhanh"},
        ]
        service.ensure_ready = AsyncMock()
        service._materialize_images = AsyncMock(return_value=[{"token": "img"}])
        service.check_history = AsyncMock(
            return_value=local_app.build_history_state(
                cid="",
                status="missing",
                message="missing",
                checked=True,
                saved=False,
            )
        )

        payload = local_app.ChatRequest(
            prompt="Generate an image of a red paper lantern.",
            model="gemini-3-pro-advanced",
            timeout_seconds=150,
        )

        events = []
        async for event in service.stream_chat_events(payload):
            events.append(event)

        self.assertTrue(fake_chat.calls[0][1]["use_pro"])
        self.assertEqual(service.current_model, "gemini-3-pro-advanced")
        final_event = next(event for event in events if event.get("type") == "final")
        self.assertEqual(final_event["imageMode"], "Image generation Pro")

    async def test_image_prompt_without_pro_stays_on_standard_image_path(self):
        service = local_app.GeminiLocalService()
        standard_chat = FakeChat(FakeOutput(text="done", images=[object()]))
        fake_client = SimpleNamespace(timeout=30.0, watchdog_timeout=30.0, start_chat=lambda model: standard_chat)

        service.client = fake_client
        service.chat = standard_chat
        service.current_model = "gemini-3-flash"
        service.ensure_ready = AsyncMock()
        service._materialize_images = AsyncMock(return_value=[{"token": "img"}])
        service.check_history = AsyncMock(
            return_value=local_app.build_history_state(
                cid="",
                status="missing",
                message="missing",
                checked=True,
                saved=False,
            )
        )

        payload = local_app.ChatRequest(
            prompt="Generate an image of a red paper lantern.",
            model="gemini-3-flash",
            timeout_seconds=150,
        )

        result = await service.ask(payload)

        self.assertEqual(result["imageMode"], "Image generation")
        self.assertEqual(len(standard_chat.calls), 1)
        self.assertFalse(standard_chat.calls[0][1]["use_pro"])

    async def test_stream_image_prompt_without_pro_stays_on_standard_image_path(self):
        service = local_app.GeminiLocalService()
        standard_chat = FakeChat(FakeOutput(text="done", images=[object()]))
        fake_client = SimpleNamespace(timeout=30.0, watchdog_timeout=30.0, start_chat=lambda model: standard_chat)

        service.client = fake_client
        service.chat = standard_chat
        service.current_model = "gemini-3-flash"
        service.ensure_ready = AsyncMock()
        service._materialize_images = AsyncMock(return_value=[{"token": "img"}])
        service.check_history = AsyncMock(
            return_value=local_app.build_history_state(
                cid="",
                status="missing",
                message="missing",
                checked=True,
                saved=False,
            )
        )

        payload = local_app.ChatRequest(
            prompt="Generate an image of a red paper lantern.",
            model="gemini-3-flash",
            timeout_seconds=150,
        )

        events = []
        async for event in service.stream_chat_events(payload):
            events.append(event)

        self.assertEqual(len(standard_chat.calls), 1)
        self.assertFalse(standard_chat.calls[0][1]["use_pro"])
        final_event = next(event for event in events if event.get("type") == "final")
        self.assertEqual(final_event["imageMode"], "Image generation")
        self.assertEqual(final_event["images"], [{"token": "img"}])

    async def test_materialized_cached_images_include_download_metadata(self):
        service = local_app.GeminiLocalService()
        request_dir = local_app.IMAGE_CACHE_DIR / "unit-test"
        request_dir.mkdir(parents=True, exist_ok=True)
        saved_path = request_dir / "20260410_abcd_image_0.png"
        saved_path.write_bytes(b"png")

        try:
            payload = await service._materialize_images(
                [FakeSavedImage(str(saved_path))],
                "unit-test",
            )
        finally:
            if saved_path.exists():
                saved_path.unlink()
            if request_dir.exists():
                request_dir.rmdir()

        self.assertEqual(len(payload), 1)
        image = payload[0]
        self.assertEqual(
            image["proxyUrl"],
            "/generated/unit-test/20260410_abcd_image_0.png",
        )
        self.assertEqual(
            image["downloadUrl"],
            "/generated/unit-test/20260410_abcd_image_0.png",
        )
        self.assertEqual(image["downloadName"], "20260410_abcd_image_0.png")
        self.assertEqual(image["quality"], "full")

    async def test_generated_images_use_server_managed_urls(self):
        service = local_app.GeminiLocalService()
        service.client = make_fake_client()
        request_dir = local_app.IMAGE_CACHE_DIR / "generated-unit"

        try:
            with patch.object(local_app, "GeneratedImage", FakeGeneratedImage):
                image = FakeGeneratedImage(
                    url="https://lh3.googleusercontent.com/gg-dl/token",
                    preview_url="https://lh3.googleusercontent.com/gg-dl/token",
                    title="[Generated Image 0]",
                    alt="red-lantern",
                    cid="cid",
                    rid="rid",
                    rcid="rcid",
                    image_id="image-id",
                )
                payload = await service._materialize_images([image], "generated-unit")

            self.assertEqual(len(payload), 1)
            generated = payload[0]
            self.assertTrue(generated["proxyUrl"].startswith("/api/generated-image/"))
            self.assertTrue(generated["downloadUrl"].endswith("?download=true"))
            self.assertEqual(generated["quality"], "fhd")
            self.assertTrue(generated["cached"])

            token = generated["token"]
            record = service.generated_image_records[token]
            self.assertIsNotNone(record["cachedRelativePath"])
            self.assertEqual(record["quality"], "fhd")
        finally:
            if request_dir.exists():
                for child in request_dir.iterdir():
                    if child.is_file():
                        child.unlink()
                request_dir.rmdir()

    async def test_deferred_generated_images_return_preview_first(self):
        service = local_app.GeminiLocalService()
        service.client = make_fake_client()

        def fake_start_cache(token, delay_seconds=0.0):
            service.generated_image_records[token]["cacheStatus"] = "caching"

        async def fake_resolve_browser_url(record):
            record["browserUrl"] = "https://lh3.googleusercontent.com/rd-gg-dl/token=s2048"
            return record["browserUrl"]

        with (
            patch.object(service, "_start_generated_image_cache", side_effect=fake_start_cache) as start_cache,
            patch.object(
                service,
                "_resolve_generated_browser_url",
                AsyncMock(side_effect=fake_resolve_browser_url),
            ),
        ):
            image = FakeGeneratedImage(
                url="https://lh3.googleusercontent.com/gg-dl/token",
                preview_url="https://lh3.googleusercontent.com/gg-dl/token",
                title="[Generated Image 0]",
                alt="red-lantern",
                cid="cid",
                rid="rid",
                rcid="rcid",
                image_id="image-id",
            )
            payload = await service._materialize_images(
                [image],
                "generated-preview",
                defer_generated_cache=True,
            )

        self.assertEqual(len(payload), 1)
        generated = payload[0]
        self.assertTrue(generated["serverManaged"])
        self.assertFalse(generated["cached"])
        self.assertFalse(generated["downloadReady"])
        self.assertTrue(generated["previewReady"])
        self.assertFalse(generated["localPreviewReady"])
        self.assertEqual(
            generated["browserUrl"],
            "https://lh3.googleusercontent.com/rd-gg-dl/token=s2048",
        )
        self.assertTrue(generated["proxyUrl"].startswith("/api/generated-image/"))
        self.assertIsNone(generated["downloadUrl"])
        self.assertEqual(generated["quality"], "pending")
        self.assertEqual(generated["cacheStatus"], "caching")
        start_cache.assert_called_once_with(generated["token"], delay_seconds=1.5)

    async def test_reprobe_auth_is_blocked_while_session_is_busy(self):
        service = local_app.GeminiLocalService()
        service.client = SimpleNamespace(configured_cookie_fingerprint="stale")
        service.runtime_cookie_values = {
            "__Secure-1PSID": "psid",
            "__Secure-1PSIDTS": "psidts",
        }
        service.runtime_cookie_source = "runtime paste"

        async with service._hold_session_activity("generated image save", "HiggsFlow"):
            reprobed = await service.reprobe_auth(force=True)

        self.assertFalse(reprobed)
        self.assertTrue(
            any("HiggsFlow generated image save" in entry["message"] for entry in local_app.LOG_BUFFER)
        )

    def test_infer_request_source_detects_higgsflow(self):
        request = SimpleNamespace(
            headers={
                "user-agent": "HiggsFlow/1.0",
                "origin": "https://higgsflow.app",
            },
            client=SimpleNamespace(host="10.0.0.8"),
        )

        self.assertEqual(local_app.infer_request_source(request), "HiggsFlow")

    async def test_sync_request_auto_reauths_once_after_suspended_stream_error(self):
        service = local_app.GeminiLocalService()
        abort_chat = FakeAbortChat()
        recovered_chat = FakeChat()
        fake_client = SimpleNamespace(timeout=30.0, watchdog_timeout=30.0)

        service.client = fake_client
        service.chat = abort_chat
        service.current_model = "gemini-3-flash"
        service.ensure_ready = AsyncMock()
        service.check_history = AsyncMock(
            return_value=local_app.build_history_state(
                cid="",
                status="missing",
                message="missing",
                checked=True,
                saved=False,
            )
        )

        async def fake_reprobe(force=False):
            service.chat = recovered_chat
            return True

        service.reprobe_auth = AsyncMock(side_effect=fake_reprobe)

        payload = local_app.ChatRequest(
            prompt="Say hello after auth refresh.",
            model="gemini-3-flash",
            timeout_seconds=60,
        )

        result = await service.ask(payload)

        self.assertEqual(result["text"], "done")
        self.assertEqual(len(abort_chat.calls), 1)
        self.assertEqual(len(recovered_chat.calls), 1)
        service.reprobe_auth.assert_awaited_once_with(force=True)
        self.assertTrue(
            any(
                "Re-authenticating the session and retrying this request once." in entry["message"]
                for entry in local_app.LOG_BUFFER
            )
        )

    async def test_stream_request_auto_reauths_once_after_suspended_stream_error(self):
        service = local_app.GeminiLocalService()
        abort_chat = FakeAbortChat()
        recovered_chat = FakeChat()
        fake_client = SimpleNamespace(timeout=30.0, watchdog_timeout=30.0)

        service.client = fake_client
        service.chat = abort_chat
        service.current_model = "gemini-3-flash"
        service.ensure_ready = AsyncMock()
        service.check_history = AsyncMock(
            return_value=local_app.build_history_state(
                cid="",
                status="missing",
                message="missing",
                checked=True,
                saved=False,
            )
        )

        async def fake_reprobe(force=False):
            service.chat = recovered_chat
            return True

        service.reprobe_auth = AsyncMock(side_effect=fake_reprobe)

        payload = local_app.ChatRequest(
            prompt="Say hello after stream auth refresh.",
            model="gemini-3-flash",
            timeout_seconds=60,
        )

        events = []
        async for event in service.stream_chat_events(payload):
            events.append(event)

        self.assertEqual(len(abort_chat.calls), 1)
        self.assertEqual(len(recovered_chat.calls), 1)
        service.reprobe_auth.assert_awaited_once_with(force=True)
        self.assertTrue(any(event.get("message") == local_app.AUTO_REAUTH_STATUS_MESSAGE for event in events))
        self.assertTrue(
            any(
                event.get("message") == "Re-auth completed. Retrying the same Gemini request once."
                for event in events
            )
        )
        final_event = next(event for event in events if event.get("type") == "final")
        self.assertEqual(final_event["text"], "done")

    async def test_generated_image_endpoint_recaches_from_record(self):
        service = local_app.GeminiLocalService()
        service.client = make_fake_client()
        request_dir = local_app.IMAGE_CACHE_DIR / "generated-endpoint"

        try:
            with patch.object(local_app, "GeneratedImage", FakeGeneratedImage):
                image = FakeGeneratedImage(
                    url="https://lh3.googleusercontent.com/gg-dl/token",
                    preview_url="https://lh3.googleusercontent.com/gg-dl/token",
                    title="[Generated Image 0]",
                    alt="red-lantern",
                    cid="cid",
                    rid="rid",
                    rcid="rcid",
                    image_id="image-id",
                )
                record = service._register_generated_image(image, "generated-endpoint", 0)
                response = await service.get_generated_image_response(
                    record["token"],
                    download=True,
                )

            self.assertIsInstance(response, local_app.FileResponse)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(record["quality"], "fhd")
            self.assertIsNotNone(record["cachedRelativePath"])
            cached_path = local_app.IMAGE_CACHE_DIR / record["cachedRelativePath"]
            self.assertTrue(cached_path.exists())
        finally:
            if request_dir.exists():
                for child in request_dir.iterdir():
                    if child.is_file():
                        child.unlink()
                request_dir.rmdir()

    async def test_generated_image_endpoint_proxies_browser_preview_before_cache_ready(self):
        service = local_app.GeminiLocalService()
        proxy_client = FakeProxyClient(
            [FakeProxyResponse(200, "image/png", content=b"png")]
        )
        service.client = make_fake_client(proxy_client)
        image = FakeGeneratedImage(
            url="https://lh3.googleusercontent.com/gg-dl/token",
            preview_url="https://lh3.googleusercontent.com/gg-dl/token",
            title="[Generated Image 0]",
            alt="red-lantern",
            cid="cid",
            rid="rid",
            rcid="rcid",
            image_id="image-id",
        )
        record = service._register_generated_image(image, "generated-preview-endpoint", 0)
        record["browserUrl"] = "https://lh3.googleusercontent.com/rd-gg-dl/token=s2048"

        response = await service.get_generated_image_response(record["token"], download=False)

        self.assertIsInstance(response, local_app.Response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.media_type, "image/png")
        self.assertEqual(response.body, b"png")
        self.assertEqual(proxy_client.urls, [record["browserUrl"]])

    async def test_generated_browser_url_prefers_s2048_suffix(self):
        service = local_app.GeminiLocalService()
        record = {
            "previewUrl": "https://lh3.googleusercontent.com/gg-dl/token",
            "sourceUrl": "https://lh3.googleusercontent.com/gg-dl/token",
            "browserUrl": None,
            "requestTag": "generated-browser",
            "requestSource": "local-ui",
        }

        browser_url = await service._resolve_generated_browser_url(record)

        self.assertEqual(
            browser_url,
            "https://lh3.googleusercontent.com/gg-dl/token=s2048",
        )
        self.assertEqual(record["browserUrl"], browser_url)

    def test_generated_payload_marks_browser_preview_as_ready(self):
        service = local_app.GeminiLocalService()
        record = {
            "token": "preview-test",
            "title": "[Generated Image 0]",
            "alt": "red-lantern",
            "previewUrl": "https://lh3.googleusercontent.com/gg-dl/token",
            "sourceUrl": "https://lh3.googleusercontent.com/gg-dl/token",
            "browserUrl": "https://lh3.googleusercontent.com/rd-gg-dl/token=s2048",
            "downloadName": "test.png",
            "quality": "pending",
            "cacheStatus": "caching",
            "error": None,
            "previewRelativePath": None,
            "cachedRelativePath": None,
        }

        payload = service._build_generated_image_payload(record)

        self.assertTrue(payload["previewReady"])
        self.assertFalse(payload["localPreviewReady"])
        self.assertTrue(payload["proxyUrl"].startswith("/api/generated-image/preview-test"))
        self.assertEqual(payload["browserUrl"], record["browserUrl"])

    async def test_generated_browser_url_uses_s4096_for_pro_mode(self):
        service = local_app.GeminiLocalService()
        record = {
            "previewUrl": "https://lh3.googleusercontent.com/gg-dl/token",
            "sourceUrl": "https://lh3.googleusercontent.com/gg-dl/token",
            "browserUrl": None,
            "requestTag": "generated-browser-pro",
            "requestSource": "local-ui",
            "preferredPreviewSize": 4096,
        }

        browser_url = await service._resolve_generated_browser_url(record)

        self.assertEqual(
            browser_url,
            "https://lh3.googleusercontent.com/gg-dl/token=s4096",
        )

    async def test_generated_image_download_does_not_redirect_to_browser_url_after_save_failure(self):
        service = local_app.GeminiLocalService()
        service.client = make_fake_client()
        image = FakeGeneratedImage(
            url="https://lh3.googleusercontent.com/gg-dl/token",
            preview_url="https://lh3.googleusercontent.com/gg-dl/token",
            title="[Generated Image 0]",
            alt="red-lantern",
            cid="cid",
            rid="rid",
            rcid="rcid",
            image_id="image-id",
        )
        record = service._register_generated_image(image, "generated-download-failure", 0)
        record["browserUrl"] = "https://lh3.googleusercontent.com/rd-gg-dl/token=s2048"

        with patch.object(service, "_cache_generated_image_record", AsyncMock(side_effect=RuntimeError("Error downloading image: 403"))):
            with self.assertRaises(local_app.HTTPException) as ctx:
                await service.get_generated_image_response(record["token"], download=True)

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("local save failed", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()

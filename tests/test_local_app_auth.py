import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import local_app  # noqa: E402
from gemini_webapi.constants import AccountStatus  # noqa: E402


def cookie_blob(cookies: dict[str, str]) -> str:
    return "\n".join(f"{name}={value}" for name, value in cookies.items())


def make_models() -> list[dict[str, str]]:
    return [{"name": "gemini-3-flash", "label": "Gemini 3 Flash"}]


class TestLocalAppAuth(unittest.IsolatedAsyncioTestCase):
    async def test_status_reprobes_when_configured_fingerprint_changes(self):
        service = local_app.GeminiLocalService()
        old_cookies = {
            "__Secure-1PSID": "old-psid",
            "__Secure-1PSIDTS": "old-ts",
        }
        new_cookies = {
            "__Secure-1PSID": "new-psid",
            "__Secure-1PSIDTS": "new-ts",
        }

        service.client = SimpleNamespace(
            configured_cookie_fingerprint=local_app.build_cookie_fingerprint(old_cookies),
            auth_source="explicit",
            auth_cookie_fingerprint="old-auth",
            account_status=AccountStatus.AVAILABLE,
        )
        service.current_model = "gemini-3-flash"
        service.available_models = make_models()
        service.account_status = AccountStatus.AVAILABLE
        service._sync_cookie_snapshot(old_cookies, "cookies.json + .env")

        async def fake_initialize(cookies, source):
            service.client = SimpleNamespace(
                configured_cookie_fingerprint=local_app.build_cookie_fingerprint(cookies),
                auth_source="explicit",
                auth_cookie_fingerprint="new-auth",
                account_status=AccountStatus.AVAILABLE,
            )
            service.current_model = "gemini-3-flash"
            service.available_models = make_models()
            service.account_status = AccountStatus.AVAILABLE
            service.chat = object()
            service.boot_error = None
            service._sync_cookie_snapshot(cookies, source)

        with (
            patch.object(local_app, "load_configured_cookies", return_value=(new_cookies, "cookies.json + .env")),
            patch.object(service, "_initialize_locked", AsyncMock(side_effect=fake_initialize)) as init_mock,
            patch.object(service, "close", AsyncMock()),
        ):
            data = await service.status()

        self.assertTrue(data["ready"])
        self.assertTrue(data["reprobed"])
        self.assertEqual(data["credentials"]["fingerprint"], local_app.build_cookie_fingerprint(new_cookies))
        self.assertEqual(data["activeAuth"]["authSource"], "explicit")
        self.assertTrue(data["activeAuth"]["matchesConfigured"])
        init_mock.assert_awaited_once()

    async def test_runtime_paste_snapshot_takes_priority_over_disk_source(self):
        service = local_app.GeminiLocalService()
        runtime_cookies = {
            "__Secure-1PSID": "runtime-psid",
            "__Secure-1PSIDTS": "runtime-ts",
            "SID": "runtime-sid",
        }
        disk_cookies = {
            "__Secure-1PSID": "disk-psid",
            "__Secure-1PSIDTS": "disk-ts",
        }

        async def fake_initialize(cookies, source):
            service.client = SimpleNamespace(
                configured_cookie_fingerprint=local_app.build_cookie_fingerprint(cookies),
                auth_source="explicit",
                auth_cookie_fingerprint="runtime-auth",
                account_status=AccountStatus.AVAILABLE,
            )
            service.current_model = "gemini-3-flash"
            service.available_models = make_models()
            service.account_status = AccountStatus.AVAILABLE
            service.chat = object()
            service.boot_error = None
            service._sync_cookie_snapshot(cookies, source)

        init_mock = AsyncMock(side_effect=fake_initialize)

        with patch.object(service, "_initialize_locked", init_mock), patch.object(service, "close", AsyncMock()):
            payload = await service.update_credentials(
                local_app.CookieUpdateRequest(
                    raw_cookies=cookie_blob(runtime_cookies),
                    persist=False,
                )
            )

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["credentials"]["source"], "runtime paste")
            self.assertEqual(payload["activeAuth"]["authSource"], "explicit")

            with patch.object(local_app, "load_configured_cookies", return_value=(disk_cookies, "cookies.json + .env")):
                data = await service.status()

        self.assertTrue(data["ready"])
        self.assertFalse(data["reprobed"])
        self.assertEqual(data["credentials"]["source"], "runtime paste")
        self.assertTrue(data["activeAuth"]["matchesConfigured"])
        self.assertEqual(init_mock.await_count, 1)


if __name__ == "__main__":
    unittest.main()

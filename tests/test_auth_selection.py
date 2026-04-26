import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from curl_cffi.requests import Cookies

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from gemini_webapi.constants import AccountStatus  # noqa: E402

get_access_token_module = importlib.import_module("gemini_webapi.utils.get_access_token")


class FakeResponse:
    def __init__(self, status_code: int = 200, text: str = ""):
        self.status_code = status_code
        self.text = text


class FakeSession:
    def __init__(self, *args, **kwargs):
        self.cookies = Cookies()
        self.timeout = None

    async def get(self, url, headers=None):
        return FakeResponse(status_code=200, text="")

    async def close(self):
        return None


def build_cache_payload(cookies: dict[str, str]) -> str:
    return json.dumps(
        [
            {
                "name": name,
                "value": value,
                "domain": ".google.com",
                "path": "/",
            }
            for name, value in cookies.items()
        ]
    )


async def fake_send_request(client, cookies, verbose=False):
    client.cookies.clear()
    if isinstance(cookies, Cookies):
        client.cookies.update(cookies)
    else:
        for name, value in cookies.items():
            client.cookies.set(name, value, domain=".google.com")

    return FakeResponse(
        status_code=200,
        text=(
            '"SNlM0e":"token",'
            '"cfb2h":"build-label",'
            '"FdrFJe":"session-id",'
            '"TuX5cc":"en",'
            '"qKIAYe":"push-id"'
        ),
    )


class TestAuthSelection(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_cookies_are_tested_before_cache_when_psid_pair_matches(self):
        explicit_cookies = {
            "__Secure-1PSID": "same-psid",
            "__Secure-1PSIDTS": "same-ts",
            "EXPLICIT_MARKER": "1",
        }
        cache_cookies = {
            "__Secure-1PSID": "same-psid",
            "__Secure-1PSIDTS": "same-ts",
            "CACHE_MARKER": "1",
        }
        probes: list[str] = []

        async def fake_probe(client, access_token, build_label, session_id, language, verbose=False):
            if client.cookies.get("EXPLICIT_MARKER"):
                probes.append("explicit")
                return AccountStatus.UNAUTHENTICATED
            if client.cookies.get("CACHE_MARKER"):
                probes.append("cache")
                return AccountStatus.AVAILABLE
            self.fail("Unexpected cookie candidate probed.")

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / ".cached_cookies_same.json"
            cache_file.write_text(build_cache_payload(cache_cookies), encoding="utf-8")

            with (
                patch.object(get_access_token_module, "AsyncSession", FakeSession),
                patch.object(get_access_token_module, "_send_request", fake_send_request),
                patch.object(get_access_token_module, "_probe_account_status", fake_probe),
                patch.object(get_access_token_module, "_get_cookies_cache_path", return_value=cache_file),
                patch.object(get_access_token_module, "load_browser_cookies", return_value={}),
            ):
                *_, session = await get_access_token_module.get_access_token(explicit_cookies)

        self.assertEqual(probes, ["explicit", "cache"])
        self.assertEqual(session._gemini_auth_source, "cache")
        self.assertEqual(session._gemini_auth_account_status, AccountStatus.AVAILABLE)

    async def test_explicit_available_wins_before_cache(self):
        explicit_cookies = {
            "__Secure-1PSID": "same-psid",
            "__Secure-1PSIDTS": "same-ts",
            "EXPLICIT_MARKER": "1",
        }
        cache_cookies = {
            "__Secure-1PSID": "same-psid",
            "__Secure-1PSIDTS": "same-ts",
            "CACHE_MARKER": "1",
        }
        probes: list[str] = []

        async def fake_probe(client, access_token, build_label, session_id, language, verbose=False):
            if client.cookies.get("EXPLICIT_MARKER"):
                probes.append("explicit")
                return AccountStatus.AVAILABLE
            if client.cookies.get("CACHE_MARKER"):
                probes.append("cache")
                return AccountStatus.AVAILABLE
            self.fail("Unexpected cookie candidate probed.")

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / ".cached_cookies_same.json"
            cache_file.write_text(build_cache_payload(cache_cookies), encoding="utf-8")

            with (
                patch.object(get_access_token_module, "AsyncSession", FakeSession),
                patch.object(get_access_token_module, "_send_request", fake_send_request),
                patch.object(get_access_token_module, "_probe_account_status", fake_probe),
                patch.object(get_access_token_module, "_get_cookies_cache_path", return_value=cache_file),
                patch.object(get_access_token_module, "load_browser_cookies", return_value={}),
            ):
                *_, session = await get_access_token_module.get_access_token(explicit_cookies)

        self.assertEqual(probes, ["explicit"])
        self.assertEqual(session._gemini_auth_source, "explicit")
        self.assertEqual(session._gemini_auth_account_status, AccountStatus.AVAILABLE)

    async def test_browser_fallback_runs_after_explicit_failure(self):
        explicit_cookies = {
            "__Secure-1PSID": "same-psid",
            "__Secure-1PSIDTS": "explicit-ts",
        }
        browser_cookies = {
            "chrome": [
                {
                    "name": "__Secure-1PSID",
                    "value": "same-psid",
                    "domain": ".google.com",
                    "path": "/",
                },
                {
                    "name": "__Secure-1PSIDTS",
                    "value": "browser-ts",
                    "domain": ".google.com",
                    "path": "/",
                },
            ]
        }

        async def fake_probe(client, access_token, build_label, session_id, language, verbose=False):
            if client.cookies.get("__Secure-1PSIDTS") == "explicit-ts":
                return AccountStatus.UNAUTHENTICATED
            if client.cookies.get("__Secure-1PSIDTS") == "browser-ts":
                return AccountStatus.AVAILABLE
            self.fail("Unexpected cookie candidate probed.")

        with (
            patch.object(get_access_token_module, "AsyncSession", FakeSession),
            patch.object(get_access_token_module, "_send_request", fake_send_request),
            patch.object(get_access_token_module, "_probe_account_status", fake_probe),
            patch.object(get_access_token_module, "_get_cookies_cache_path", return_value=None),
            patch.object(get_access_token_module, "load_browser_cookies", return_value=browser_cookies),
        ):
            *_, session = await get_access_token_module.get_access_token(explicit_cookies)

        self.assertEqual(session._gemini_auth_source, "browser:chrome")
        self.assertEqual(session._gemini_auth_account_status, AccountStatus.AVAILABLE)


if __name__ == "__main__":
    unittest.main()

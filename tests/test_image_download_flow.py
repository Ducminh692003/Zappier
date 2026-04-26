import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from gemini_webapi.types.image import GeneratedImage, Image  # noqa: E402


class FakeResponse:
    def __init__(self, status_code: int, content_type: str, text: str = "", content: bytes = b"", reason: str = "OK"):
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.text = text
        self.content = content
        self.reason = reason


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.urls = []

    async def get(self, url, headers=None):
        self.urls.append(url)
        if not self._responses:
            raise AssertionError("No more fake responses configured.")
        return self._responses.pop(0)


class FakeClientRef:
    def __init__(self, original_url: str):
        self.original_url = original_url

    async def _get_full_size_image(self, **kwargs):
        return self.original_url


class TestImageDownloadFlow(unittest.IsolatedAsyncioTestCase):
    async def test_base_image_save_follows_text_redirect_chain(self):
        image = Image(url="https://lh3.googleusercontent.com/gg-dl/token")
        client = FakeClient(
            [
                FakeResponse(200, "text/plain", text="https://lh3.google.com/rd-gg/token"),
                FakeResponse(200, "text/plain", text="https://lh3.googleusercontent.com/rd-gg/token"),
                FakeResponse(200, "image/png", content=b"png-bytes"),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_path = await image._perform_save(client, Path(temp_dir), "sample", verbose=False)

            self.assertTrue(Path(saved_path).exists())
            self.assertEqual(Path(saved_path).read_bytes(), b"png-bytes")
            self.assertEqual(
                client.urls,
                [
                    "https://lh3.googleusercontent.com/gg-dl/token",
                    "https://lh3.google.com/rd-gg/token",
                    "https://lh3.googleusercontent.com/rd-gg/token",
                ],
            )

    async def test_generated_image_save_uses_size_suffix_without_rpc(self):
        image = GeneratedImage(
            url="https://lh3.googleusercontent.com/gg-dl/image_token",
            preview_url="https://lh3.googleusercontent.com/gg-dl/image_token",
            client_ref=None,
            cid="",
            rid="",
            rcid="",
            image_id="",
        )
        client = FakeClient([FakeResponse(200, "image/png", content=b"png-bytes")])

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_path = await image._perform_save(client, Path(temp_dir), "generated", verbose=False, full_size=True)

            self.assertTrue(Path(saved_path).exists())
            self.assertIn("=s2048", client.urls[0])
            self.assertEqual(image.saved_quality, "fhd")

    async def test_generated_image_save_uses_s4096_for_pro_preview_size(self):
        image = GeneratedImage(
            url="https://lh3.googleusercontent.com/gg-dl/image_token",
            preview_url="https://lh3.googleusercontent.com/gg-dl/image_token",
            client_ref=None,
            cid="",
            rid="",
            rcid="",
            image_id="",
            preferred_preview_size=4096,
        )
        client = FakeClient([FakeResponse(200, "image/png", content=b"png-bytes")])

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_path = await image._perform_save(client, Path(temp_dir), "generated", verbose=False, full_size=True)

            self.assertTrue(Path(saved_path).exists())
            self.assertIn("=s4096", client.urls[0])
            self.assertEqual(image.saved_quality, "max")

    async def test_generated_image_save_falls_back_to_preview_when_full_size_fails(self):
        image = GeneratedImage(
            url="https://lh3.googleusercontent.com/gg-dl/image_token",
            preview_url="https://lh3.googleusercontent.com/gg-dl/image_token",
            client_ref=FakeClientRef("https://lh3.googleusercontent.com/fife/full_size_token"),
            cid="cid",
            rid="rid",
            rcid="rcid",
            image_id="image-id",
        )
        client = FakeClient(
            [
                FakeResponse(403, "text/html", reason="Forbidden"),
                FakeResponse(403, "text/html", reason="Forbidden"),
                FakeResponse(200, "text/plain", text="https://work.fife.usercontent.google.com/rd-gg-dl/token"),
                FakeResponse(200, "image/png", content=b"png-bytes"),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_path = await image._perform_save(
                client,
                Path(temp_dir),
                "generated",
                verbose=False,
                full_size=True,
            )

            self.assertTrue(Path(saved_path).exists())
            self.assertEqual(
                client.urls,
                [
                    "https://lh3.googleusercontent.com/fife/full_size_token=d-I?alr=yes",
                    "https://lh3.googleusercontent.com/gg-dl/image_token=s2048",
                    "https://lh3.googleusercontent.com/gg-dl/image_token",
                    "https://work.fife.usercontent.google.com/rd-gg-dl/token",
                ],
            )
            self.assertEqual(image.saved_quality, "preview")

    async def test_generated_preview_prefetch_prefers_fhd_candidate(self):
        image = GeneratedImage(
            url="https://lh3.googleusercontent.com/gg-dl/image_token",
            preview_url="https://lh3.googleusercontent.com/gg-dl/image_token",
            client_ref=None,
            cid="",
            rid="",
            rcid="",
            image_id="",
        )
        client = FakeClient(
            [
                FakeResponse(200, "text/plain", text="https://work.fife.usercontent.google.com/rd-gg-dl/preview-token"),
                FakeResponse(200, "image/png", content=b"png-bytes"),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_path = await image._perform_save(
                client,
                Path(temp_dir),
                "generated-preview",
                verbose=False,
                full_size=False,
            )

            self.assertTrue(Path(saved_path).exists())
            self.assertEqual(
                client.urls,
                [
                    "https://lh3.googleusercontent.com/gg-dl/image_token=s2048",
                    "https://work.fife.usercontent.google.com/rd-gg-dl/preview-token",
                ],
            )
            self.assertEqual(image.saved_quality, "preview")


if __name__ == "__main__":
    unittest.main()

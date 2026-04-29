import hashlib
import mimetypes
from datetime import datetime
from pathlib import Path
from textwrap import shorten
from typing import Any

from curl_cffi.requests import AsyncSession
from curl_cffi.requests.exceptions import HTTPError
from pydantic import BaseModel, ConfigDict

from ..constants import Headers
from ..utils.http_session import create_gemini_session
from ..utils import logger


class Image(BaseModel):
    """
    A single image object returned from Gemini.

    Parameters
    ----------
    url: `str`
        URL of the image.
    title: `str`, optional
        Title of the image, defaults to "[Image]".
    alt: `str`, optional
        Optional description of the image.
    proxy: `str`, optional
        Proxy used when saving image.
    client: `AsyncSession`, optional
        Used for saving file with authentication if needed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str
    title: str = "[Image]"
    alt: str = ""
    proxy: str | None = None
    client: AsyncSession | None = None
    saved_quality: str | None = None
    _default_filename_suffix: str = "image"

    def _get_url_for_hash(self) -> str:
        return self.url

    def __repr__(self) -> str:
        return f"Image(title={self.title!r}, alt={shorten(self.alt, width=100)!r}, url={self.url!r})"

    async def save(
        self,
        path: str = "temp",
        filename: str | None = None,
        verbose: bool = False,
        client: AsyncSession | None = None,
        **kwargs,
    ) -> str:
        """
        Saves the image to disk.

        Parameters
        ----------
        path: `str`, optional
            Directory path to save the image, defaults to "./temp".
        filename: `str | None`, optional
            File name to save the image. Defaults to a unique generated name.
        verbose: `bool`, optional
            If True, will print the path of the saved file or warning for invalid file name. Defaults to False.
        client: `AsyncSession | None`, optional
            Client used for requests.
        kwargs: `dict`, optional
            Additional arguments passed to the specific image's `_perform_save` implementation.
            For example, `GeneratedImage` accepts `full_size (bool)`.

        Returns
        -------
        `str`
            Absolute path of the saved image if successful.

        Raises
        ------
        `curl_cffi.requests.exceptions.HTTPError`
            If the network request failed.
        """

        if not filename or not Path(filename).suffix:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            url_hash = hashlib.sha256(self._get_url_for_hash().encode()).hexdigest()[
                :10
            ]
            base_name = (
                Path(filename).stem if filename else self._default_filename_suffix
            )
            filename = f"{timestamp}_{url_hash}_{base_name}"

        close_client = False
        req_client = client or self.client
        if not req_client:
            client_ref = getattr(self, "client_ref", None)
            shared_client = getattr(client_ref, "client", None) if client_ref else None
            if shared_client:
                req_client = shared_client
            else:
                cookies = getattr(client_ref, "cookies", None) if client_ref else None
                req_client = create_gemini_session(
                    proxy=self.proxy,
                    cookies=cookies,
                )
                close_client = True

        try:
            path_obj = Path(path)
            path_obj.mkdir(parents=True, exist_ok=True)
            return await self._perform_save(
                req_client, path_obj, filename, verbose, **kwargs
            )
        finally:
            if close_client:
                await req_client.close()

    async def _perform_save(
        self, req_client: AsyncSession, path_obj: Path, filename: str, verbose: bool
    ) -> str:
        """
        Base implementation: simple download.
        """
        current_url = self.url
        response = None
        request_headers = dict(Headers.REFERER.value)

        for _ in range(8):
            response = await req_client.get(current_url, headers=request_headers)
            if verbose:
                logger.debug(f"HTTP Request: GET {current_url} [{response.status_code}]")

            if response.status_code != 200:
                break

            content_type = (
                response.headers.get("content-type", "").split(";")[0].strip().lower()
            )
            if content_type.startswith("image/"):
                path_obj_file = Path(filename)
                if not path_obj_file.suffix:
                    ext = mimetypes.guess_extension(content_type) or ".png"
                    filename = f"{filename}{ext}"

                dest = path_obj / filename
                dest.write_bytes(response.content)

                if verbose:
                    logger.info(f"Image saved as {dest.resolve()}")

                self.saved_quality = self.saved_quality or "source"
                return str(dest.resolve())

            if content_type.startswith("text/plain"):
                next_url = response.text.strip()
                if next_url.startswith("http"):
                    current_url = next_url
                    self.url = current_url
                    continue

            break

        raise HTTPError(
            f"Error downloading image: {response.status_code if response else 'N/A'} "
            f"{getattr(response, 'reason', '') if response else ''}".strip()
        )


class WebImage(Image):
    """
    Image retrieved from web.

    Returned when asking Gemini to "SEND an image of [something]".
    """

    pass


class GeneratedImage(Image):
    """
    Image generated by Gemini.

    Returned when asking Gemini to "GENERATE an image of [something]".

    Parameters
    ----------
    client_ref: `GeminiClient`, optional
        Reference to the GeminiClient instance.
    cid: `str`, optional
        Chat ID.
    rid: `str`, optional
        Reply ID.
    rcid: `str`, optional
        Reply candidate ID.
    image_id: `str`, optional
        Image ID generated.
    """

    client_ref: Any = None
    cid: str = ""
    rid: str = ""
    rcid: str = ""
    image_id: str = ""
    preview_url: str = ""
    preferred_preview_size: int = 2048

    @staticmethod
    def _build_base_candidate(url: str) -> str | None:
        if not url or "googleusercontent.com" not in url:
            return None
        return url.split("=", 1)[0]

    @classmethod
    def _build_handoff_candidate(cls, url: str) -> str | None:
        base = cls._build_base_candidate(url)
        if not base:
            return None
        return f"{base}=d-I?alr=yes"

    @classmethod
    def _build_full_size_candidate(cls, url: str) -> str | None:
        base = cls._build_base_candidate(url)
        if not base:
            return None
        return f"{base}=s0-d?alr=yes"

    @classmethod
    def _build_full_size_candidates(cls, url: str) -> list[str]:
        candidates: list[str] = []

        def add(candidate: str | None) -> None:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        add(cls._build_full_size_candidate(url))
        add(cls._build_handoff_candidate(url))
        add(url)
        return candidates

    @classmethod
    def _build_scaled_candidate(cls, url: str, size: int) -> str | None:
        base = cls._build_base_candidate(url)
        if not base:
            return None
        return f"{base}=s{max(1, int(size))}"

    @classmethod
    def _build_fhd_candidate(cls, url: str) -> str | None:
        return cls._build_scaled_candidate(url, 2048)

    def _build_preferred_preview_candidate(self, url: str) -> str | None:
        size = 4096 if int(self.preferred_preview_size or 0) >= 4096 else 2048
        return self._build_scaled_candidate(url, size)

    def _build_preview_candidates(
        self,
        url: str,
        *,
        prefer_fhd: bool,
    ) -> list[str]:
        candidates: list[str] = []

        def add(candidate: str | None) -> None:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        if prefer_fhd:
            add(self._build_preferred_preview_candidate(url))
        add(url)
        return candidates

    # @override
    async def _perform_save(
        self,
        req_client: AsyncSession,
        path_obj: Path,
        filename: str,
        verbose: bool,
        full_size: bool = True,
    ) -> str:
        """
        Internal method for saving GeneratedImage, handling full size resolution.

        Parameters
        ----------
        req_client: `AsyncSession`
             Client used for requests.
        path_obj: `Path`
            Path to save the image.
        filename: `str`
            Base filename.
        verbose: `bool`
            Prints status if True.
        full_size: `bool`, optional
            Modifies preview URLs to fetch full-size images. Defaults to True.

        Returns
        -------
        `str`
            Absolute path of the saved image if successfully saved.
        """

        preview_url = self.preview_url or self.url
        self.preview_url = preview_url
        self.saved_quality = None

        if full_size:
            if all([self.client_ref, self.cid, self.rid, self.rcid, self.image_id]):
                try:
                    original_url = await self.client_ref._get_full_size_image(
                        cid=self.cid,
                        rid=self.rid,
                        rcid=self.rcid,
                        image_id=self.image_id,
                    )
                    if original_url:
                        for candidate in self._build_full_size_candidates(original_url):
                            self.url = candidate
                            try:
                                saved_path = await super()._perform_save(
                                    req_client, path_obj, filename, verbose
                                )
                                self.saved_quality = "full"
                                return saved_path
                            except HTTPError as e:
                                self.url = preview_url
                                logger.debug(
                                    f"Failed to download Gemini full-size image: {e}. Trying the next full-size source."
                                )

                except Exception as e:
                    self.url = preview_url
                    logger.debug(
                        f"Failed to fetch full size image URL via RPC: {e}, falling back to default URL suffix."
                    )

            fhd_url = self._build_preferred_preview_candidate(preview_url)
            if fhd_url and fhd_url != preview_url:
                try:
                    self.url = fhd_url
                    saved_path = await super()._perform_save(
                        req_client, path_obj, filename, verbose
                    )
                    self.saved_quality = (
                        "max" if int(self.preferred_preview_size or 0) >= 4096 else "fhd"
                    )
                    return saved_path
                except HTTPError as e:
                    self.url = preview_url
                    logger.debug(
                        f"Failed to download Gemini FHD image via preview fallback: {e}. Falling back to plain preview."
                    )

            last_error: HTTPError | None = None
            self.url = preview_url
            try:
                saved_path = await super()._perform_save(
                    req_client, path_obj, filename, verbose
                )
                self.saved_quality = "preview"
                return saved_path
            except HTTPError as e:
                last_error = e
                logger.debug(
                    f"Failed to download a Gemini preview fallback: {e}."
                )

            if last_error is not None:
                raise last_error
            raise HTTPError("Error downloading image: preview fallback was unavailable")
        else:
            preview_candidates = self._build_preview_candidates(
                preview_url,
                prefer_fhd=True,
            )

            last_error: HTTPError | None = None
            for candidate in preview_candidates:
                self.url = candidate
                try:
                    saved_path = await super()._perform_save(
                        req_client, path_obj, filename, verbose
                    )
                    self.saved_quality = "preview"
                    return saved_path
                except HTTPError as e:
                    last_error = e
                    logger.debug(
                        f"Failed to prefetch a Gemini preview fallback: {e}. Trying the next preview source."
                    )

            if last_error is not None:
                raise last_error
            raise HTTPError("Error downloading image: preview fallback was unavailable")

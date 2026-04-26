class AuthError(Exception):
    """
    Exception for authentication errors caused by invalid credentials/cookies.
    """

    pass


class APIError(Exception):
    """
    Exception for package-level errors which need to be fixed in the future development (e.g. validation errors).
    """

    pass


class ImageGenerationError(APIError):
    """
    Exception for generated image parsing errors.
    """

    pass


class StreamSuspendedError(Exception):
    """
    Exception for a Gemini stream that stopped before a recoverable chat id was available.
    """

    def __init__(
        self,
        message: str,
        *,
        completed: bool | None = None,
        final_chunk: bool | None = None,
        thinking: bool | None = None,
        queueing: bool | None = None,
        request_id: int | str | None = None,
    ):
        super().__init__(message)
        self.completed = completed
        self.final_chunk = final_chunk
        self.thinking = thinking
        self.queueing = queueing
        self.request_id = request_id


class GeminiError(Exception):
    """
    Exception for errors returned from Gemini server which are not handled by the package.
    """

    pass


class TimeoutError(GeminiError):
    """
    Exception for request timeouts.
    """

    pass


class UsageLimitExceeded(GeminiError):
    """
    Exception for model usage limit exceeded errors.
    """

    pass


class ModelInvalid(GeminiError):
    """
    Exception for invalid model header string errors.
    """

    pass


class TemporarilyBlocked(GeminiError):
    """
    Exception for 429 Too Many Requests when IP is temporarily blocked.
    """

    pass

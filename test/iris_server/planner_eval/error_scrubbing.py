"""Keep a secret out of the evaluation's output.

An API error can echo part of the key it rejected, even masked. So nothing the
evaluation prints or saves ever contains an exception's message: only its class
name and, when it carries one, its HTTP status code.
"""


class ModelCallFailed(Exception):
    """A model call failed; the message is the scrubbed description, nothing more."""


def http_status_of(error):
    """The HTTP status an API error carries, or None."""
    for source in (error, getattr(error, "response", None)):
        status = getattr(source, "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool):
            return status
    return None


def describe_error(error: BaseException) -> str:
    """Class name and HTTP status only, e.g. "RateLimitError (HTTP 429)"."""
    if isinstance(error, ModelCallFailed):
        return str(error)
    status = http_status_of(error)
    name = type(error).__name__
    return f"{name} (HTTP {status})" if status is not None else name


def scrubbed(ask):
    """`ask` with every failure replaced by a ModelCallFailed that holds no message text."""

    def ask_scrubbed(*arguments):
        try:
            return ask(*arguments)
        except Exception as error:
            # `from None` also drops the original from the traceback.
            raise ModelCallFailed(describe_error(error)) from None

    return ask_scrubbed


def exit_with_scrubbed_error(main) -> None:
    """Run `main`; if it fails, exit with the scrubbed description instead of a traceback."""
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:
        raise SystemExit(f"evaluation stopped: {describe_error(error)}") from None

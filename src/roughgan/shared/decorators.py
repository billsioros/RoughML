import inspect
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


def benchmark(method):
    """The following decorator aims at calculating the decorated function's execution
    time and is used to benchmark our various approaches and assist us in coming up with
    a comprehensive comparison of their efficiency."""

    @wraps(method)
    def wrapper(*args, **kwargs):
        beg = time.time()
        rv = method(*args, **kwargs)
        end = time.time()

        logger.info("%s returned after %7.3f seconds", method.__name__, end - beg)

        return rv

    return wrapper


def debug(method):
    """The following decorator serves at emitting details regarding the decorated
    function's calls.

    In more detai, the information emitted is:
        - The function's name.
        - Its positional and keyword arguments for the function call at hand.
        - Any exception that the function `raises`.

    In addition to that, the `debug` decorator passes a special boolean keyword argument
    by the name `debug`, if and only if it is included in the function signature.
    You can then utilize this argument inside the decorated function and emit additional
    information.
    """
    signature = inspect.signature(method)

    defaults = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    @wraps(method)
    def wrapper(*args, **kwargs):
        called_with = ""
        if args:
            called_with += ", ".join(str(x) for x in args)
            called_with += ", "

        called_with += ", ".join(f"{x}={kwargs.get(x, defaults[x])}" for x in defaults)

        try:
            rv = method(*args, **kwargs)
        except Exception as e:
            logger.debug("%s(%s) raised %s", method.__name__, called_with, e)
            raise

        logger.debug("%s(%s) returned %s", method.__name__, called_with, rv)

        return rv

    return wrapper

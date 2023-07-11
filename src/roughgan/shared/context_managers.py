class ExceptionLoggingHandler:
    def __init__(self, logger, suppress_exceptions=True) -> None:
        self.logger = logger
        self.suppress_exceptions = suppress_exceptions

    def __enter__(self):
        self._success = True

        return self

    def __exit__(self, exc_type, exc_value, _):
        if exc_value is not None and exc_type != KeyboardInterrupt:
            self._success = False
            self.logger.exception(exc_value)

            return self.suppress_exceptions
        return None

    @property
    def success(self):
        return self._success

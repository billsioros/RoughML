class Configuration:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = Configuration(**value)

            setattr(self, key, value)

    def to_dict(self):
        rv = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Configuration):
                value = value.to_dict()

            rv[key] = value

        return rv

    def __str__(self) -> str:
        return str(self.to_dict())

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} '{self!s}'>"

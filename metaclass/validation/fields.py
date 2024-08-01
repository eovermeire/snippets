from abc import ABC, abstractmethod
import re


class BaseField(ABC):
    """Base class for creating custom fields."""

    @abstractmethod
    def validate(self) -> bool:
        """"Validate the custom Field."""
        raise NotImplementedError


class URL(BaseField):
    """URL Field."""
    def __init__(self, url: str) -> None:
        self.url = url

    def __repr__(self) -> str:
        return self.url

    def validate(self) -> bool:
        """Simple URL validation."""
        url_regex = re.compile(r'https?://(?:www\.)?[a-zA-Z0-9./]+')
        return bool(url_regex.match(self.url))

from unittest import TestCase

from validation.model import BaseModel
from validation.fields import URL
from validation.exceptions import ValidationError

class TestModel(BaseModel):
    int_var: int
    float_var: float
    url: URL

class TestBaseModel(TestCase):
    """Test the BaseModel class."""

    def test_valid_model(self) -> None:
        model = TestModel(int_var=42, float_var=3.14, url=URL("http://127.0.0.1"))
        
        assert model.int_var == 42
        assert model.float_var == 3.14
        assert model.url.url == "http://127.0.0.1"

    def test_raise_type_error(self) -> None:
        with self.assertRaises(TypeError):
            _ = TestModel(int_var=42, float_var="3.14", url=URL("http://127.0.0.1"))

    def test_raise_attribute_error_extra(self) -> None:
        with self.assertRaises(AttributeError):
            _ = TestModel(int_var=42, float_var=3.14, urk=URL("http://127.0.0.1"))

    def test_raise_attribute_error_missing(self) -> None:
        with self.assertRaises(AttributeError):
            _ = TestModel(int_var=42, float_var=3.14)

    def test_raise_validation_error(self) -> None:
        with self.assertRaises(ValidationError):
            _ = TestModel(int_var=42, float_var=3.14, url=URL("I am not a url"))
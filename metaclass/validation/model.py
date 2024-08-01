from inspect import get_annotations

from validation.exceptions import ValidationError
from validation.fields import BaseField

class BaseMeta(type):
    """Creates validated BaseModels."""
    def __new__(cls, name, bases, dct):
        def show(self) -> str:
            attrs_str = ", ".join([f"{k} = {v}" for k, v in self.__dict__.items()])
            return f"{type(self).__name__}({attrs_str})"

        dct["__repr__"] = show
        new_cls = super().__new__(cls, name, bases, dct)

        return new_cls

    def __call__(cls, *args, **kwargs):
        annotations = get_annotations(cls, eval_str=True)
        instance = super().__call__(*args, **kwargs)

        for k, v in annotations.items():
            if k not in kwargs:
                raise AttributeError(f"missing attribute {k} of type {v}")

            if type(kwargs[k]) is not v:
                raise TypeError(f"type of {k} is not {v}")

            if issubclass(type(kwargs[k]), BaseField) and not kwargs[k].validate():
                raise ValidationError(f"{k} failed to validate")

            setattr(instance, k, kwargs[k])

        return instance


class BaseModel(metaclass=BaseMeta):
    """Base class for custom models."""
    def __init__(self, *args, **kwargs):
        pass
from typing import Generic, TypeVar, Optional
from pydantic.generics import GenericModel

T = TypeVar("T")


class ResponseModel(GenericModel, Generic[T]):
    code: int = 0
    msg: str = "success"
    data: Optional[T] = None


def response_success(msg = "success", data = None):
    return ResponseModel(code=200, msg=msg, data=data)


def response_error(msg="error"):
    return ResponseModel(code=500, msg=msg)

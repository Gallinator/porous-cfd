from pydantic import BaseModel


class Predict2dInput(BaseModel):
    uuid: str
    model: str
    points: dict[str, list[float]]


class Response2d(BaseModel):
    points: dict[str, list[float]]
    target: dict[str, list[float]]
    predicted: dict[str, list[float]]
    error: dict[str, list[float]]
    residuals: dict[str, list[float]]
    porous_ids: list[float] | None = None

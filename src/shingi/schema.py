"""Request validation for the documented TypeSafe HTTP contract."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

Description = str | dict[str, JsonValue] | list[JsonValue]


class QuestionBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    instructions: Description


class Choice(QuestionBase):
    type: Literal["choice"]
    criteria: dict[str, Description | None] = Field(min_length=1, max_length=255)


class Score(QuestionBase):
    type: Literal["score"]
    criteria: list[Description] = Field(min_length=2, max_length=10)


class Noul(QuestionBase):
    type: Literal["noul"]
    criteria: dict[str, Description] | None = None

    @field_validator("criteria")
    @classmethod
    def known_keys(cls, value):
        if value is not None and set(value) - {"true", "false"}:
            raise ValueError("noul criteria keys must be true or false")
        return value


Question = Annotated[Choice | Score | Noul, Field(discriminator="type")]


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str
    state: Description
    questions: dict[str, Question] = Field(min_length=1, max_length=256)

    @field_validator("state")
    @classmethod
    def text_only(cls, value):
        if isinstance(value, dict):
            for key in ("image", "screenshot"):
                image = value.get(key)
                if isinstance(image, str) and (image.startswith("data:image") or len(image) > 2000):
                    raise ValueError("image input is not supported; Shingi 27B is text only")
        return value

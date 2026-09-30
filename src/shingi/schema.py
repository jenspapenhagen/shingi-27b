"""Request validation for the documented TypeSafe HTTP contract and the SGLang decisions route."""
import base64
import binascii
import unicodedata
from typing import Annotated, Any, Literal

from pydantic import (AfterValidator, BaseModel, ConfigDict, Discriminator, Field, JsonValue, Tag,
                      field_validator, model_validator)

from .decision import MEDIA_MARKER, describe

Description = str | dict[str, JsonValue] | list[JsonValue]
MAX_IMAGES = 8
MAX_IMAGE_BYTES = 20 * 1024 * 1024
# Formats the runtime decodes (stb_image) without extra tools.
IMAGE_SIGNATURES = {b"\x89PNG\r\n\x1a\n": "PNG", b"\xff\xd8\xff": "JPEG", b"GIF87a": "GIF", b"GIF89a": "GIF",
                    b"BM": "BMP"}


def image_base64(value):
    """Validate one image given as a data:image/...;base64 URL or plain base64; return plain base64."""
    payload = value
    if value.startswith("data:"):
        header, _, payload = value.partition(",")
        if not header.startswith("data:image/") or not header.endswith(";base64"):
            raise ValueError("an image data URL must be data:image/<type>;base64,...")
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("an image must be base64 image bytes or a base64 data URL") from exc
    if not data:
        raise ValueError("an image must not be empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"an image must be at most {MAX_IMAGE_BYTES // (1024 * 1024)} MiB")
    if not any(data.startswith(signature) for signature in IMAGE_SIGNATURES):
        raise ValueError("unsupported image format; send PNG, JPEG, GIF or BMP")
    # Canonical padded base64 for the native readout's strict decoder.
    return base64.b64encode(data).decode("ascii")


Image = Annotated[str, AfterValidator(image_base64)]
Images = Annotated[list[Image], Field(max_length=MAX_IMAGES)]


def reject_marker_text(*texts):
    """With images, the text must not contain the media marker, or images would land in the wrong place."""
    if any(MEDIA_MARKER in describe(text) for text in texts):
        raise ValueError(f"text must not contain the media marker {MEDIA_MARKER} when images are attached")


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


def question_texts(question):
    criteria = question.criteria or []
    if isinstance(criteria, dict):
        return [question.instructions, *criteria.keys(), *criteria.values()]
    return [question.instructions, *criteria]


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str
    state: Description
    questions: dict[str, Question] = Field(min_length=1, max_length=256)
    # Shingi extension: base64 images or data URLs, placed in order before the text.
    images: Images | None = None

    @field_validator("state")
    @classmethod
    def no_inline_images(cls, value):
        if isinstance(value, dict):
            for key in ("image", "screenshot"):
                image = value.get(key)
                if isinstance(image, str) and (image.startswith("data:image") or len(image) > 2000):
                    raise ValueError("inline image data in state is not read; send images in the top-level images list")
        return value

    @model_validator(mode="after")
    def markers_only_for_images(self):
        if self.images:
            texts = [self.state]
            for question in self.questions.values():
                texts += question_texts(question)
            reject_marker_text(*texts)
        return self


# SGLang /v1/decisions, System One shaped body (sgl-project/sglang#41712): a state and keyed questions.
class StateDecisionRequest(Request):
    model: str | None = None
    questions: dict[str, Question] = Field(min_length=1, max_length=16)
    # Divides the candidate logits after calibration; the decision is unchanged.
    temperature: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    thinking: dict[str, JsonValue] | None = None

    @field_validator("thinking")
    @classmethod
    def no_thinking(cls, value):
        if value and value.get("enabled"):
            raise ValueError("the thinking handoff is not served")
        return value


# SGLang /v1/decisions, generic body: an input and a list of typed questions with ids.
def nonblank(value):
    if not describe(value).strip():
        raise ValueError("must not be blank")
    return value


Text = Annotated[Description, AfterValidator(nonblank)]


class Option(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: Description | None = None


class GenericQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Annotated[str, AfterValidator(nonblank)]
    question: Text


class ChoiceQuestion(GenericQuestion):
    type: Literal["choice"]
    options: list[Option] = Field(min_length=2, max_length=26)

    @field_validator("options")
    @classmethod
    def distinct_names(cls, options):
        seen = set()
        for option in options:
            key = option.name.strip().casefold()
            if not key:
                raise ValueError("option names must be nonempty")
            if any(unicodedata.category(c) in ("Cc", "Zl", "Zp") for c in option.name):
                raise ValueError(f"option name {option.name!r} must not contain control or line break characters")
            if key in seen:
                raise ValueError(f"option name {option.name!r} repeats another option")
            seen.add(key)
        return options


class ScoreQuestion(GenericQuestion):
    type: Literal["score"]
    levels: list[Text] = Field(min_length=2, max_length=10)


class YesNoQuestion(GenericQuestion):
    type: Literal["yes_no"]
    yes: Description | None = None
    no: Description | None = None


GenericDecisionQuestion = Annotated[ChoiceQuestion | ScoreQuestion | YesNoQuestion, Field(discriminator="type")]
PROMPT_FORMAT_VERSION = 1


class InputDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: Text
    questions: list[GenericDecisionQuestion] = Field(min_length=1, max_length=256)
    temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    prompt_format_version: int | None = None
    return_prompt_token_ids: bool = False
    model: str | None = None
    # Shingi extension: base64 images or data URLs, placed in order before the text.
    images: Images | None = None

    @field_validator("questions")
    @classmethod
    def distinct_ids(cls, questions):
        seen = set()
        for question in questions:
            if question.id in seen:
                raise ValueError(f"question id {question.id!r} repeats another question")
            seen.add(question.id)
        return questions

    @field_validator("chat_template_kwargs")
    @classmethod
    def thinking_off(cls, value):
        if set(value) - {"enable_thinking"} or value.get("enable_thinking", False) is not False:
            raise ValueError("only enable_thinking=false is supported; decisions never think")
        return value

    @field_validator("prompt_format_version")
    @classmethod
    def served_version(cls, value):
        if value is not None and value != PROMPT_FORMAT_VERSION:
            raise ValueError(f"this server serves prompt_format_version {PROMPT_FORMAT_VERSION}")
        return value

    @field_validator("return_prompt_token_ids")
    @classmethod
    def no_token_ids(cls, value):
        if value:
            raise ValueError("return_prompt_token_ids is not supported")
        return value

    @model_validator(mode="after")
    def markers_only_for_images(self):
        if self.images:
            texts = [self.input]
            for q in self.questions:
                texts.append(q.question)
                if isinstance(q, ChoiceQuestion):
                    texts += [x for o in q.options for x in (o.name, o.description)]
                elif isinstance(q, ScoreQuestion):
                    texts += q.levels
                else:
                    texts += [q.yes, q.no]
            reject_marker_text(*texts)
        return self


def decisions_body_kind(body):
    # The generic body has an input and a list of questions, so the shapes never overlap.
    if isinstance(body, dict) and ("state" in body or isinstance(body.get("questions"), dict)):
        return "state"
    return "input"


DecisionsRequest = Annotated[
    Annotated[StateDecisionRequest, Tag("state")] | Annotated[InputDecisionRequest, Tag("input")],
    Discriminator(decisions_body_kind),
]

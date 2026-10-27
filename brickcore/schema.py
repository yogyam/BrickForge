"""Authoritative build contract (schema_version 1.0). Python schemas are the source of truth."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"
COORDINATE_SYSTEM = "stud_xy_plate_z_v1"
BRICK_HEIGHT_PLATES = 3


class Bounds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x_studs: int = Field(ge=1, le=24)
    y_studs: int = Field(ge=1, le=24)
    z_plates: int = Field(ge=3, le=36)

    @model_validator(mode="after")
    def _multiple_of_brick(self) -> "Bounds":
        if self.z_plates % BRICK_HEIGHT_PLATES != 0:
            raise ValueError("z_plates must be a multiple of 3 in R1")
        return self

    @property
    def layers(self) -> int:
        return self.z_plates // BRICK_HEIGHT_PLATES


class Constraints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_pieces: int = Field(ge=1, le=100)
    bounds: Bounds
    color_id: str


class GeneratorInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    version: str
    seed: int
    strategy: str | None = None


class Placement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    part_key: str
    x: int
    y: int
    z: int
    rotation_deg: Literal[0, 90]
    color_id: str


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=1)
    added_placement_ids: list[str]


class Build(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    build_id: str
    catalog_version: str
    coordinate_system: Literal["stud_xy_plate_z_v1"] = COORDINATE_SYSTEM
    generator: GeneratorInfo
    constraints: Constraints
    placements: list[Placement]
    steps: list[Step]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "Build":
        return cls.model_validate_json(text)


ResultStatus = Literal["validated_candidate", "partial_candidate", "no_valid_candidate"]

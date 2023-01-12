"""Typing for wbfw109 packages"""
import dataclasses
import datetime
from typing import (
    Any,
    Callable,
    Generic,
    MutableMapping,
    MutableSequence,
    NamedTuple,
    Optional,
    ParamSpec,
    Tuple,
    TypeAlias,
    TypedDict,
    TypeVar,
    Union,
)

# Title: Generic ~
# if "T" is annotated on "cls" keyword, "T" indicates a class.
# else if "type[T]" is annotated on "cls" keyword, "T" indicates a object.
T = TypeVar("T")
AnyT = TypeVar("AnyT")
P = ParamSpec("P")
# Data Structure Type like dataclasses.dataclass
DST = TypeVar("DST", bound=object)

# Title: Generic with Typed
JsonPrimitiveValueType: TypeAlias = Optional[bool | int | float | str]
JsonType: TypeAlias = MutableMapping[
    str,
    Union["JsonType", JsonPrimitiveValueType | MutableSequence[JsonPrimitiveValueType]],
]
RecursiveTuple: TypeAlias = tuple[Union["RecursiveTuple", Any], ...]


@dataclasses.dataclass
class SingleLinkedList(Generic[T, AnyT]):
    node: T
    target: AnyT


# Title: Composition type ~
class JudgeResult(NamedTuple):
    elapsed_time: float
    judge_result: bool | Any


# Title: decorator ~
def deprecated(t: T) -> T:
    """It will be used until PEP 702 is implemented.
    - PEP 702 – Marking deprecations using the type system ; https://peps.python.org/pep-0702/
    """
    return t


# Title: ref algorithms~ ~
class ReferenceJsonAlgorithmDetailType(TypedDict):
    season: str
    round: str
    name: str
    dataset_hyperlink: str
    is_in_resources: bool


class ReferenceJsonAlgorithmMetaType(TypedDict):
    company: str
    contest: str


# Title: ref.toml ~
class ReferenceTomlAlgorithmsType(TypedDict):
    python_packages_path: str
    parent_path_order: list[str]
    problem_path_order: list[str]


class ReferenceTomlProjectType(TypedDict):
    resources_root: str
    src_root: str


class ReferenceTomlType(TypedDict):
    project: ReferenceTomlProjectType
    algorithms: ReferenceTomlAlgorithmsType


# outdated ~
TimeEdgePair = Tuple[datetime.time, datetime.time]
# float value is total seconds.
TimeRange = Tuple[datetime.time, float]
DatetimeRange = Tuple[datetime.datetime, float]


class Point2D(NamedTuple):
    x: float
    y: float


class RangeOfTwoPoint(NamedTuple):
    start: int
    end: int

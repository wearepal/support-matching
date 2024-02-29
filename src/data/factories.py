"""Dataset factories."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing_extensions import override

from conduit.data.datasets.vision import NICOPP, NicoPPTarget
from conduit.fair.data.datasets import (
    ACSDataset,
    ACSHorizon,
    ACSSetting,
    ACSState,
    ACSSurvey,
    ACSSurveyYear,
)

from src.data.common import DatasetFactory

__all__ = ["NICOPPCfg"]


@dataclass
class NICOPPCfg(DatasetFactory):
    root: Path | str
    target_attrs: list[NicoPPTarget] | None = None
    transform: Any = None  # Optional[Union[Compose, BasicTransform, Callable[[Image], Any]]]

    @override
    def __call__(self) -> NICOPP:
        return NICOPP(root=self.root, transform=self.transform, superclasses=self.target_attrs)


@dataclass
class ACSCfg(DatasetFactory):
    setting: ACSSetting
    survey_year: ACSSurveyYear = ACSSurveyYear.YEAR_2018
    horizon: ACSHorizon = ACSHorizon.ONE_YEAR
    survey: ACSSurvey = ACSSurvey.PERSON
    states: list[ACSState] = field(default_factory=lambda: [ACSState.AL])

    @override
    def __call__(self) -> ACSDataset:
        return ACSDataset(
            setting=self.setting,
            survey_year=self.survey_year,
            horizon=self.horizon,
            survey=self.survey,
            states=self.states,
        )

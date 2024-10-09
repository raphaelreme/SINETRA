"""Code and config to run trackmate"""

import dataclasses

import pathlib
import os

from byotrack.implementation.linker import trackmate


@dataclasses.dataclass
class TrackMateConfig(trackmate.TrackMateParameters):
    """Configuration for KOFT algorithm"""

    fiji_path = pathlib.Path(os.environ.get("FIJI", ""))

    def build(self) -> trackmate.TrackMateLinker:
        return trackmate.TrackMateLinker(self.fiji_path, self)

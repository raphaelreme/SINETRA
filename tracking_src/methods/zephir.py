"""Code and config to run ZephIR"""

import dataclasses
import os
import pathlib
import sys

import numpy as np
import torch

import byotrack

sys.path.append(f"{os.environ.get('EXPYRUN_CWD', '.')}/ZephIR/")

# pylint: disable=import-error,wrong-import-position
from zephir.methods import build_annotations, build_models, build_springs, build_tree, track_all  # type: ignore
from zephir.methods.recommend_frames import recommend_frames  # type: ignore
from zephir.models.container import Container  # type: ignore


@dataclasses.dataclass
class ZephIRParameters:
    allow_rotation: bool = True
    gamma: float = 1.0  # No need to correct video
    grid_step: int = 11  # Neurons are small cf paper
    nn_max: int = 5  # Max number of neighbors
    sort_mode: str = "linear"
    lambda_t: float = -1.0
    lambda_d: float = 1.0
    lambda_n: float = 0.1
    lambda_n_mode: str = "norm"
    lr_floor: float = 0.08
    lr_ceiling: float = 0.15
    n_epoch: int = 20  # Number of epochs, 20 seems enough to converge with the IR loss
    n_epoch_d: int = 20  # Number of epochs with Det loss, 20 seems enough also


class ZephIRLinker(byotrack.Linker):
    """Use the official ZephIR implementation to produce tracks.

    It is able to track without detections (but detections does improve the results)

    This linker will only works with our modification to ZephIR code and with our synthetic data. Some variation
    could be made to fully integrate ZephIR in ByoTrack format.

    It automatically extracts annotated frames from the ground truth (using zephIR `recommend_frames`
    to choose the frames). It currently does not support to run with 0 annotated frame. (We could give
    the detections on the most recommended frame as annotation.)
    """

    def __init__(
        self, specs: ZephIRParameters, dataset: pathlib.Path, num_annotated_frames=1, device="cpu", verbose=False
    ):
        super().__init__()
        self.dataset = dataset
        self.device = device
        self.specs = specs
        self.verbose = verbose

        # Recommed frames for the given dataset (This is part of the annotation process
        # and does not really belong to the algorithmic part)
        recommend_frames(
            dataset=dataset,
            n_frames=num_annotated_frames,
            n_iter=-1,
            t_list=None,
            channel=0,
            save_to_metadata=False,
            verbose=self.verbose,
        )

    def run(self, video, detections_sequence):
        dim = detections_sequence[0].dim
        # defining variable container with some key arguments
        container = Container(
            dataset=self.dataset,
            allow_rotation=self.specs.allow_rotation,
            channel=0,
            dev=self.device,
            exclude_self=True,
            exclusive_prov=None,
            gamma=self.specs.gamma,
            include_all=True,
            n_frame=1,
            z_compensator=-1,
            lr_coef=2.0,
        )

        # loading and handling annotations
        container, results = build_annotations(
            container=container,
            annotation=None,
            t_ref=None,
            wlid_ref=None,
            n_ref=None,
        )

        # compiling models
        container, zephir, _ = build_models(
            container=container,
            dimmer_ratio=0.1,
            grid_shape=(1 if dim == 2 else self.specs.grid_step, self.specs.grid_step, self.specs.grid_step),
            fovea_sigma=(1 if dim == 2 else self.specs.grid_step, 2.5, 2.5),
            n_chunks=1,
        )

        # compiling spring network
        container = build_springs(
            container=container,
            load_nn=True,
            nn_max=self.specs.nn_max,
            verbose=self.verbose,
        )

        # building frame tree
        container = build_tree(
            container=container,
            sort_mode=self.specs.sort_mode,
            t_ignore=None,
            t_track=list(range(0, len(video))),
            verbose=False,
        )

        # tracking: It will fill results tensor
        container, results = track_all(
            container=container,
            results=results,
            zephir=zephir,
            detections_sequence=detections_sequence,
            clip_grad=1.0,
            lambda_t=self.specs.lambda_t,
            lambda_d=self.specs.lambda_d,
            lambda_n=self.specs.lambda_n,
            lambda_n_mode=self.specs.lambda_n_mode,
            lr_ceiling=self.specs.lr_ceiling,
            lr_floor=self.specs.lr_floor,
            motion_predict=True,
            n_epoch=self.specs.n_epoch,
            n_epoch_d=self.specs.n_epoch_d,
            _t_list=None,
        )

        # Convert tracks to the right format
        tracks = []

        points = (
            (1 + results[..., ::-1])
            / 2
            * np.array([container.metadata["shape_z"], container.metadata["shape_y"], container.metadata["shape_x"]])
        )

        offset = container.metadata["shape_z"] == 1  # 2d vs 3d

        for i in range(results.shape[1]):
            tracks.append(byotrack.Track(start=0, points=torch.tensor(points[:, i, offset:].astype(np.float32))))

        return tracks


@dataclasses.dataclass
class ZephIRConfig(ZephIRParameters):
    dataset: pathlib.Path = pathlib.Path("")
    device: str = "cpu"
    num_annotated_frames: int = 1

    def build(self) -> ZephIRLinker:
        linker = ZephIRLinker(self, self.dataset, self.num_annotated_frames, self.device)
        return linker

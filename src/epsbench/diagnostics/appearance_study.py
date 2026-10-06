"""Two immutable source contracts; caller objects/configurations confer no authority."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, cast

from epsbench.utils.canonical import sha256_bytes

ExecutionVersion = Literal[
    "paired_appearance_execution_v1", "appearance_contrast_calibration_execution_v1"
]
NativePurpose = Literal["paired_appearance_native_v1", "appearance_contrast_calibration_native_v1"]
DummyPurpose = Literal["paired_appearance_dummy_v1", "appearance_contrast_calibration_dummy_v1"]


@dataclass(frozen=True, slots=True)
class StudySpec:
    schema: str
    config_path: str
    config_file_bytes: bytes
    appearances: tuple[str, str]
    palette: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]
    asset_directory: str
    asset_names: tuple[str, ...]
    asset_hashes: tuple[str, ...]
    asset_pixel_hashes: tuple[str, ...]
    execution_version: ExecutionVersion
    native_purpose: NativePurpose
    dummy_purpose: DummyPurpose

    @property
    def fixed(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.config_file_bytes))

    @property
    def config_file_hash(self) -> str:
        return sha256_bytes(self.config_file_bytes)


PAIRED_V1 = StudySpec(
    "paired-appearance-development-v1",
    "configs/development/paired_appearance_v1.json",
    (
        b'{"appearances":["dev_pair_solid_v1","dev_pair_brick_v1"],"base":"0125c51'
        b'0b9d2b1e4760dcf1b987d52638bd9e18e","corridor":{"camera_lateral":0.0,"del'
        b'ta":[0.7,0.0,0.0],"floor_thickness":0.05,"fovy":55.0,"height":1.25,"leng'
        b'th":6.0,"poses":[0.5,1.2],"wall_height":5.0,"wall_thickness":0.05,"width'
        b'":3.0,"xyaxes":[1.0,0.0,0.0,0.0,0.0,1.0]},"height":120,"limits":{"endpoi'
        b'nt":1048576,"other_metadata":131072,"shared":2097152,"snapshot":65536,"s'
        b'napshots":4,"total":67108864},"palette":[[[204,132,72],[54,88,124]],[[82'
        b",184,164],[132,48,94]],[[164,104,202],[44,132,76]],[[202,184,84],[84,64,"
        b'154]]],"purpose":"development_only","repeats":2,"roots":[2026100601,2026'
        b'100602],"schema":"paired-appearance-development-v1","single":{"backgroun'
        b'd_position":[0.0,2.5,1.05],"background_size":[2.2,0.05,1.05],"camera_for'
        b'ward":-3.0,"delta":[0.0,0.7,0.0],"fovy":55.0,"height":1.25,"occluder_pos'
        b'ition":[0.0,0.8,0.9],"occluder_size":[0.55,0.05,0.9],"poses":[-0.35,0.35'
        b'],"support_size":[4.0,7.0,0.1],"xyaxes":[1.0,0.0,0.0,0.0,0.16,1.0]},"tex'
        b'ture":{"background_rule":"row%16<4 OR (column+16*((row//16)%2))%32<8","g'
        b'enerator":"staggered_brick_128_v1","resolution":128},"thresholds":{"chan'
        b'ged_fraction":0.2,"interior_pixels":100,"luminance_coefficients":[0.2126'
        b',0.7152,0.0722],"luminance_mean":[0.05,0.95],"luminance_std":0.025,"neig'
        b'hbour_difference":0.05,"neighbour_fraction":0.02,"normalized_mad":0.025}'
        b',"visual":{"ambient":[0.1,0.1,0.1],"diffuse":0.7,"directions":[[0.2,0.5,'
        b'-1.0],[0.0,0.25,-1.0]],"filtering":"mujoco_linear_mipmap_linear_v1","fog'
        b'":false,"haze":false,"offSamples":0,"positions":[[-1.0,-2.0,5.0],[0.0,-1'
        b'.0,6.0]],"reflectance":0.0,"shadows":false,"shininess":0.0,"specular":0.'
        b'0,"texrepeat":[1.0,1.0],"texuniform":false,"uv":"mujoco_geom_local_uv_re'
        b'peat_v1","zfar":[20.0,30.0],"znear":0.01},"width":160}\n'
    ),
    ("dev_pair_solid_v1", "dev_pair_brick_v1"),
    (
        ((204, 132, 72), (54, 88, 124)),
        ((82, 184, 164), (132, 48, 94)),
        ((164, 104, 202), (44, 132, 76)),
        ((202, 184, 84), (84, 64, 154)),
    ),
    "paired_appearance_v1",
    ("brick-slot-0.png", "brick-slot-1.png", "brick-slot-2.png", "brick-slot-3.png"),
    (
        "04e290cfa9313a9980374915d67f1b3750a7791e18abccafb154427c10d77b93",
        "8f6907953a943476f8dbbc83e015724f8ecd1be693eef8d6797967fab64cbc5e",
        "f3966acf038d88a33142e13c9d81d7668cfc86070e8edaa33bdc8d6d874e1464",
        "c04819b9158a6b7f0033ef963fd8e1410c3a04440606d8dda04becd9a2a65a5a",
    ),
    (
        "9b5d0c25756b0ddfa6e17e169c03c19b1435099904c341d688b9fe81a0c4cc0c",
        "87060a3a8a7ab010e55e570de38a8fe9469fc7734227e4ddba25d2130ab49782",
        "742a9bd269849b84ce39d5337606fea47e9798e121ff926911ff3d8ddfc6798f",
        "dbf364b0d4aa9370c7db1c913a9f96d2284e97303b4e91371fbb9d16b542a304",
    ),
    "paired_appearance_execution_v1",
    "paired_appearance_native_v1",
    "paired_appearance_dummy_v1",
)
CONTRAST_V1 = StudySpec(
    "appearance-contrast-calibration-v1",
    "configs/development/appearance_contrast_calibration_v1.json",
    (
        b'{"appearances":["dev_calibration_solid_v1","dev_calibration_brick_v1"],"'
        b'base":"0125c510b9d2b1e4760dcf1b987d52638bd9e18e","corridor":{"camera_lat'
        b'eral":0.0,"delta":[0.7,0.0,0.0],"floor_thickness":0.05,"fovy":55.0,"heig'
        b'ht":1.25,"length":6.0,"poses":[0.5,1.2],"wall_height":5.0,"wall_thicknes'
        b's":0.05,"width":3.0,"xyaxes":[1.0,0.0,0.0,0.0,0.0,1.0]},"height":120,"li'
        b'mits":{"endpoint":1048576,"other_metadata":131072,"shared":2097152,"snap'
        b'shot":65536,"snapshots":4,"total":67108864},"palette":[[[224,224,224],[3'
        b"2,32,32]],[[224,224,224],[32,32,32]],[[224,224,224],[32,32,32]],[[224,22"
        b'4,224],[32,32,32]]],"purpose":"development_only","repeats":2,"roots":[20'
        b'26100603,2026100604],"schema":"appearance-contrast-calibration-v1","sing'
        b'le":{"background_position":[0.0,2.5,1.05],"background_size":[2.2,0.05,1.'
        b'05],"camera_forward":-3.0,"delta":[0.0,0.7,0.0],"fovy":55.0,"height":1.2'
        b'5,"occluder_position":[0.0,0.8,0.9],"occluder_size":[0.55,0.05,0.9],"pos'
        b'es":[-0.35,0.35],"support_size":[4.0,7.0,0.1],"xyaxes":[1.0,0.0,0.0,0.0,'
        b'0.16,1.0]},"texture":{"background_rule":"row%16<4 OR (column+16*((row//1'
        b'6)%2))%32<8","generator":"staggered_brick_128_v1","resolution":128},"thr'
        b'esholds":{"changed_fraction":0.2,"interior_pixels":100,"luminance_coeffi'
        b'cients":[0.2126,0.7152,0.0722],"luminance_mean":[0.05,0.95],"luminance_s'
        b'td":0.025,"neighbour_difference":0.05,"neighbour_fraction":0.02,"normali'
        b'zed_mad":0.025},"visual":{"ambient":[0.1,0.1,0.1],"diffuse":0.7,"directi'
        b'ons":[[0.2,0.5,-1.0],[0.0,0.25,-1.0]],"filtering":"mujoco_linear_mipmap_'
        b'linear_v1","fog":false,"haze":false,"offSamples":0,"positions":[[-1.0,-2'
        b'.0,5.0],[0.0,-1.0,6.0]],"reflectance":0.0,"shadows":false,"shininess":0.'
        b'0,"specular":0.0,"texrepeat":[1.0,1.0],"texuniform":false,"uv":"mujoco_g'
        b'eom_local_uv_repeat_v1","zfar":[20.0,30.0],"znear":0.01},"width":160}\n'
    ),
    ("dev_calibration_solid_v1", "dev_calibration_brick_v1"),
    (
        ((224, 224, 224), (32, 32, 32)),
        ((224, 224, 224), (32, 32, 32)),
        ((224, 224, 224), (32, 32, 32)),
        ((224, 224, 224), (32, 32, 32)),
    ),
    "appearance_contrast_calibration_v1",
    ("brick.png",),
    ("cd83dd6fc2697cb9334d2b464166ad1e7bab96372000b2b7762e7a9b70e2f48a",),
    ("9114be5304fa8f4424eb32c137dd69fadf84da19b0f45a84fcd5f924ecaf527f",),
    "appearance_contrast_calibration_execution_v1",
    "appearance_contrast_calibration_native_v1",
    "appearance_contrast_calibration_dummy_v1",
)


def require_spec(study: StudySpec) -> StudySpec:
    if study is not PAIRED_V1 and study is not CONTRAST_V1:
        raise ValueError("exact source-allowlisted study object required")
    return study


def select_study(schema: str) -> StudySpec:
    for study in (PAIRED_V1, CONTRAST_V1):
        if schema == study.schema:
            return study
    raise ValueError("unsupported study; no arbitrary configuration authority")

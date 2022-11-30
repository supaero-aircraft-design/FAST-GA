"""Estimation of vertical tail sweeps."""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent
import fastoad.api as oad

from ..constants import SUBMODEL_VT_SWEEP


# TODO: HT and VT components are similar --> factorize
@oad.RegisterSubmodel(SUBMODEL_VT_SWEEP, "fastga.submodel.geometry.vertical_tail.sweep.legacy")
class ComputeVTSweep(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Vertical tail sweeps estimation."""

    def setup(self):
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)

        self.add_output("data:geometry:vertical_tail:sweep_0", units="deg")
        self.add_output("data:geometry:vertical_tail:sweep_50", units="rad")
        self.add_output("data:geometry:vertical_tail:sweep_100", units="deg")

        self.declare_partials(
            of="data:geometry:vertical_tail:sweep_0",
            wrt=[
                "data:geometry:vertical_tail:span",
                "data:geometry:vertical_tail:root:chord",
                "data:geometry:vertical_tail:tip:chord",
                "data:geometry:vertical_tail:sweep_25",
            ],
            method="fd",
        )
        self.declare_partials(
            of="data:geometry:vertical_tail:sweep_50",
            wrt=[
                "data:geometry:vertical_tail:span",
                "data:geometry:vertical_tail:root:chord",
                "data:geometry:vertical_tail:tip:chord",
                "data:geometry:vertical_tail:sweep_25",
                "data:geometry:vertical_tail:aspect_ratio",
                "data:geometry:vertical_tail:taper_ratio",
            ],
            method="fd",
        )
        self.declare_partials(
            of="data:geometry:vertical_tail:sweep_100",
            wrt=[
                "data:geometry:vertical_tail:span",
                "data:geometry:vertical_tail:root:chord",
                "data:geometry:vertical_tail:tip:chord",
                "data:geometry:vertical_tail:sweep_25",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        sweep_25 = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]

        sweep_0 = (
            (
                np.pi / 2
                - np.arctan2(
                    b_v,
                    (0.25 * root_chord - 0.25 * tip_chord + b_v * np.tan(sweep_25 / 180.0 * np.pi)),
                )
            )
            / np.pi
            * 180.0
        )
        sweep_50 = np.arctan(
            np.tan(sweep_0 * np.pi / 180) - 2 / ar_vt * ((1 - taper_vt) / (1 + taper_vt))
        )
        sweep_100 = (
            (
                np.pi / 2
                - np.arctan2(
                    b_v,
                    (b_v * np.tan(sweep_25 / 180.0 * np.pi) - 0.75 * root_chord + 0.75 * tip_chord),
                )
            )
            / np.pi
            * 180.0
        )

        outputs["data:geometry:vertical_tail:sweep_0"] = sweep_0
        outputs["data:geometry:vertical_tail:sweep_50"] = sweep_50
        outputs["data:geometry:vertical_tail:sweep_100"] = sweep_100

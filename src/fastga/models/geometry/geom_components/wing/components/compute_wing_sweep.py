"""Estimation of wing sweeps."""
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

import math

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent

import fastoad.api as oad

from ..constants import SUBMODEL_WING_SWEEP


@oad.RegisterSubmodel(SUBMODEL_WING_SWEEP, "fastga.submodel.geometry.wing.sweep.legacy")
class ComputeWingSweep(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Wing sweeps estimation."""

    def setup(self):

        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")

        self.add_output("data:geometry:wing:sweep_0", units="rad")
        self.add_output("data:geometry:wing:sweep_100_inner", units="rad")
        self.add_output("data:geometry:wing:sweep_100_outer", units="rad")

        self.declare_partials(
            "data:geometry:wing:sweep_0",
            [
                "data:geometry:wing:tip:leading_edge:x:local",
                "data:geometry:wing:root:y",
                "data:geometry:wing:tip:y",
            ],
            method="fd",
        )
        self.declare_partials(
            "data:geometry:wing:sweep_100_inner",
            [
                "data:geometry:wing:tip:leading_edge:x:local",
                "data:geometry:wing:root:y",
                "data:geometry:wing:tip:y",
                "data:geometry:wing:root:chord",
                "data:geometry:wing:tip:chord",
            ],
            method="fd",
        )
        self.declare_partials(
            "data:geometry:wing:sweep_100_outer",
            [
                "data:geometry:wing:tip:leading_edge:x:local",
                "data:geometry:wing:root:y",
                "data:geometry:wing:tip:y",
                "data:geometry:wing:root:chord",
                "data:geometry:wing:tip:chord",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]

        outputs["data:geometry:wing:sweep_0"] = math.atan(x4_wing / (y4_wing - y2_wing))
        outputs["data:geometry:wing:sweep_100_inner"] = math.atan(
            (x4_wing + l4_wing - l2_wing) / (y4_wing - y2_wing)
        )
        outputs["data:geometry:wing:sweep_100_outer"] = math.atan(
            (x4_wing + l4_wing - l2_wing) / (y4_wing - y2_wing)
        )

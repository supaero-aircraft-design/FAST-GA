"""Estimation of navigation systems center of gravity."""
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

from ..constants import SUBMODEL_RECORDING_SYSTEMS_CG


@oad.RegisterSubmodel(
    SUBMODEL_RECORDING_SYSTEMS_CG, "fastga.submodel.weight.cg.system.recording_system.legacy"
)
class ComputeRecordingSystemsCG(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Recording systems center of gravity estimation. Recording systems assumed to be in the tail.
    """

    def setup(self):

        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")

        self.add_output("data:weight:systems:recording:CG:x", units="m")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        lar = inputs["data:geometry:fuselage:rear_length"]
        aircraft_length = inputs["data:geometry:fuselage:length"]

        outputs["data:weight:systems:recording:CG:x"] = aircraft_length - lar / 2.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["data:weight:systems:recording:CG:x", "data:geometry:fuselage:rear_length"] = -0.5
        partials["data:weight:systems:recording:CG:x", "data:geometry:fuselage:length"] = 1.0

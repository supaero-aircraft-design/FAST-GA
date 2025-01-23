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

import fastoad.api as oad
import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent

from ..constants import SUBMODEL_RECORDING_SYSTEMS_CG


@oad.RegisterSubmodel(
    SUBMODEL_RECORDING_SYSTEMS_CG, "fastga.submodel.weight.cg.system.recording_system.legacy"
)
class ComputeRecordingSystemsCG(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Recording systems center of gravity estimation. Recording systems assumed to be in the tail.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")

        self.add_output("data:weight:systems:recording:CG:x", units="m")

        self.declare_partials(of="*", wrt="data:geometry:fuselage:length", val=1.0)
        self.declare_partials(of="*", wrt="data:geometry:fuselage:rear_length", val=-0.5)

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        lar = inputs["data:geometry:fuselage:rear_length"]
        aircraft_length = inputs["data:geometry:fuselage:length"]

        outputs["data:weight:systems:recording:CG:x"] = aircraft_length - lar / 2.0

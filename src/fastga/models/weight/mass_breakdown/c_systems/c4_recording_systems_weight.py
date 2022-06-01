"""Estimation of recording systems weight."""
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

from .constants import SUBMODEL_RECORDING_SYSTEM_MASS

oad.RegisterSubmodel.active_models[
    SUBMODEL_RECORDING_SYSTEM_MASS
] = "fastga.submodel.weight.mass.system.recording_systems.minimum"


@oad.RegisterSubmodel(
    SUBMODEL_RECORDING_SYSTEM_MASS,
    "fastga.submodel.weight.mass.system.recording_systems.minimum",
)
class ComputeRecordingSystemsWeight(ExplicitComponent):
    """
    Weight estimation for recording systems, not mandatory for airplane with weight under 5600 kg
    but the designer can add them. Indicative values are used based on the weight of DFDR and CVR
    available on the market (https://skybrary.aero/sites/default/files/bookshelf/3679.pdf)
    """

    def setup(self):

        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")

        self.add_output("data:weight:systems:recording:mass", units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mtow = inputs["data:weight:aircraft:MTOW"]

        if mtow > 5600:
            fdr_weight = 4.8
            cvr_weight = 4.5
            misc_weight = 10.0
            c4 = fdr_weight + cvr_weight + misc_weight
        else:
            c4 = 0.0

        outputs["data:weight:systems:recording:mass"] = c4

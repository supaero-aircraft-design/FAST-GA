"""
    Estimation of fuselage center of gravity.
"""
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
import logging
from openmdao.core.explicitcomponent import ExplicitComponent

import fastoad.api as oad

from ..constants import SUBMODEL_FUSELAGE_CG

_LOGGER = logging.getLogger(__name__)


@oad.RegisterSubmodel(SUBMODEL_FUSELAGE_CG, "fastga.submodel.weight.cg.airframe.fuselage.legacy")
class ComputeFuselageCG(ExplicitComponent):
    """
    Wing center of gravity estimation

    Based on a statistical analysis. See :cite:`roskampart5:1985`
    """

    def setup(self):

        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")

        self.add_output("data:weight:airframe:fuselage:CG:x", units="m")

        self.declare_partials(
            "data:weight:airframe:fuselage:CG:x", "data:geometry:fuselage:length", method="fd"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prop_layout = inputs["data:geometry:propulsion:engine:layout"]
        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]

        # Fuselage gravity center
        if prop_layout == 1.0:  # Wing mounted
            x_cg_a2 = 0.39 * fus_length
        elif prop_layout == 2.0:  # Rear fuselage mounted
            x_cg_a2 = lav + 0.485 * (fus_length - lav)
        elif prop_layout == 3.0:  # nose mount
            x_cg_a2 = lav + 0.35 * (fus_length - lav)
        else:
            _LOGGER.warning(
                "Propulsion layout %f does not exist, replaced by layout 1!", prop_layout
            )
            x_cg_a2 = 0.39 * fus_length

        outputs["data:weight:airframe:fuselage:CG:x"] = x_cg_a2

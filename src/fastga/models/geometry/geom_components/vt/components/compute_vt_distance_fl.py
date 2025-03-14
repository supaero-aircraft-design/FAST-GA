"""
Python module for the calculation of vertical tail distance from 25% wing MAC with fixed rear
fuselage length, part of the vertical tail geometry.
"""

#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
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
import openmdao.api as om
import fastoad.api as oad

from ..constants import SERVICE_VT_DISTANCE_FL, SUBMODEL_VT_DISTANCE_FL


@oad.RegisterSubmodel(SERVICE_VT_DISTANCE_FL, SUBMODEL_VT_DISTANCE_FL)
class ComputeVTMACDistanceFL(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Vertical tail mean aerodynamic chord distance estimation based on (F)ixed rear fuselage
    (L)ength (VTP distance computed).
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:at25percent:x:local", val=np.nan, units="m")
        self.add_input(
            "data:geometry:vertical_tail:MAC:at25percent:x:absolute", val=np.nan, units="m"
        )

        self.add_output("data:geometry:vertical_tail:tip:x", units="m")
        self.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", units="m")

        self.declare_partials(
            "*",
            "data:geometry:vertical_tail:MAC:at25percent:x:absolute",
            val=1.0,
        )

        self.declare_partials(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            "data:geometry:vertical_tail:MAC:at25percent:x:local",
            val=1.0,
        )

        self.declare_partials(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            "data:geometry:wing:MAC:at25percent:x",
            val=-1.0,
        )

        self.declare_partials(
            "data:geometry:vertical_tail:tip:x",
            ["data:geometry:vertical_tail:span", "data:geometry:vertical_tail:sweep_25"],
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]
        x_wing25 = inputs["data:geometry:wing:MAC:at25percent:x"]
        x_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:absolute"]
        x0_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:local"]

        vt_lp = (x_vt + x0_vt) - x_wing25
        x_tip = b_v * np.tan(sweep_25_vt) + x_vt

        outputs["data:geometry:vertical_tail:tip:x"] = x_tip
        outputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"] = vt_lp

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        partials["data:geometry:vertical_tail:tip:x", "data:geometry:vertical_tail:span"] = np.tan(
            sweep_25_vt
        )

        partials["data:geometry:vertical_tail:tip:x", "data:geometry:vertical_tail:sweep_25"] = (
            b_v / np.cos(sweep_25_vt) ** 2.0
        )

"""
Python module for the calculation of vertical tail distance from 25% wing MAC with fixed distance
between MACs, part of the vertical tail geometry.
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

from ..constants import SERVICE_VT_DISTANCE_FD, SUBMODEL_VT_DISTANCE_FD


@oad.RegisterSubmodel(SERVICE_VT_DISTANCE_FD, SUBMODEL_VT_DISTANCE_FD)
class ComputeVTMACDistanceFD(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Vertical tail mean aerodynamic chord distance estimation based on (F)ixed tail (D)istance
    from the wing.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:geometry:has_T_tail", val=np.nan)
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", units="m")

        self.declare_partials("*", "*", method="exact")

        self.declare_partials(
            "*", "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=1.0
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        has_t_tail = inputs["data:geometry:has_T_tail"]

        vt_lp = lp_ht - 0.6 * b_v * np.tan(sweep_25_vt) * has_t_tail

        outputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"] = vt_lp

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]
        has_t_tail = inputs["data:geometry:has_T_tail"]

        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            "data:geometry:vertical_tail:span",
        ] = -0.6 * np.tan(sweep_25_vt) * has_t_tail
        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            "data:geometry:vertical_tail:sweep_25",
        ] = -0.6 * b_v * has_t_tail / np.cos(sweep_25_vt) ** 2.0
        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            "data:geometry:has_T_tail",
        ] = -0.6 * np.tan(sweep_25_vt) * b_v

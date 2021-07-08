"""
    Estimation of torsion moment on fuselage section
"""

#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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
from scipy.constants import g
from fastoad.model_base import Atmosphere

FUSELAGE_MESH_POINT = 100


class ComputeTorsionMoment(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
         TASOPT
    """

    def setup(self):
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="m**2")
        self.add_input("data:TLAR:v_limit", val=np.nan, units="m/s")

        self.add_output("data:loads:fuselage:torsion_moment", units="N*m")

        self.declare_partials("data:loads:fuselage:torsion_moment", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        vtp_span = inputs["data:geometry:vertical_tail:span"]
        vtp_taper = inputs["data:geometry:vertical_tail:taper_ratio"]
        vtp_area = inputs["data:geometry:vertical_tail:area"]
        max_speed = inputs["data:TLAR:v_limit"]

        # Tail Aero Loads TODO replace Clmax with xml data and compute qNE
        rmv = 0.7
        cl_v_max = 0.2
        density = Atmosphere(0, altitude_in_feet=False).density
        q_never_exceed = 0.5 * density * max_speed ** 2
        lift_max_v = q_never_exceed * vtp_area * cl_v_max

        torsion_moment = 1 / 3 * lift_max_v * vtp_span * (1 + 2 * vtp_taper) / (1 + vtp_taper)

        outputs["data:loads:fuselage:torsion_moment"] = torsion_moment

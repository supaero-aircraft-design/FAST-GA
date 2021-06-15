"""
    Estimation of yawing moment du to the ruddder
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

import math

import numpy as np
from scipy import interpolate
from openmdao.core.explicitcomponent import ExplicitComponent
from .high_lift_aero import ComputeDeltaHighLift

from .figure_digitization import FigureDigitization


class ComputeCyDeltaRudder(FigureDigitization):
    """
    Yawing moment due to rudder estimation, rudder considered as a plian flap

    Based on : Roskam, Jan. Airplane Design: Part 6-Preliminary Calculation of Aerodynamic, Thrust and Power *
    Characteristics. DARcorporation, 1985.
    """

    def setup(self):
        self.add_input("data:aerodynamics:vertical_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:rudder:chord_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:rudder:max_deflection", val=np.nan, units="deg")

        self.add_output("data:aerodynamics:rudder:low_speed:Cy_delta_r", units="rad**-1")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        taper_ratio_vt = inputs["data:geometry:vertical_tail:taper_ratio"]
        aspect_ratio_vt = float(inputs["data:geometry:vertical_tail:aspect_ratio"])
        rudder_chord_ratio = inputs["data:geometry:vertical_tail:rudder:chord_ratio"]
        rudder_max_deflection = inputs["data:geometry:vertical_tail:rudder:max_deflection"]

        cl_alpha_vt = inputs["data:aerodynamics:vertical_tail:low_speed:CL_alpha"]

        # Assumed that the rudder covers more or less all of the vertical tail while leaving a small gap at the bottom
        # and at the top
        eta_in = 0.05
        eta_out = 0.95
        kb = self.k_b_flaps(eta_in, eta_out, taper_ratio_vt)

        # Interpolation of the first graph of figure 8.53 of Roskam
        rudder_effectiveness_parameter = self.a_delta_airfoil(rudder_chord_ratio)
        k_prime = self.k_a_delta(float(rudder_effectiveness_parameter), aspect_ratio_vt)

        # TODO: Check coherence with Roskam formula
        rudder_effectiveness_2d = self.k_prime_plain_flap(abs(rudder_max_deflection), rudder_chord_ratio)

        cy_delta_r = cl_alpha_vt * kb * k_prime * rudder_effectiveness_2d

        outputs["data:aerodynamics:rudder:low_speed:Cy_delta_r"] = cy_delta_r

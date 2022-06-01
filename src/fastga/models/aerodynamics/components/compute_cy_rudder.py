"""Estimation of yawing moment du to the rudder."""
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
import fastoad.api as oad

from .figure_digitization import FigureDigitization
from ..constants import SUBMODEL_CY_RUDDER


@oad.RegisterSubmodel(
    SUBMODEL_CY_RUDDER, "fastga.submodel.aerodynamics.rudder.yawing_moment.legacy"
)
class ComputeCyDeltaRudder(FigureDigitization):
    """
    Yawing moment due to rudder estimated based on the methodology in section 10.3.8 of Roskam
    without the surface ratio to keep the coefficient relative to the VT area and dividing by the
    theoretical airfoil lift coefficient as suggested by the formulae giving the wing lift
    increment due to flap deployment which can be considered similar

    Based on : Roskam, Jan. Airplane Design: Part 6-Preliminary Calculation of Aerodynamic,
    Thrust and Power Characteristics. DARcorporation, 1985.
    """

    def initialize(self):
        """Declaring the low_speed_aero options so we can use low speed and cruise conditions."""
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        self.add_input(
            "data:aerodynamics:vertical_tail:airfoil:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:rudder:chord_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:rudder:max_deflection", val=np.nan, units="deg")
        self.add_input("data:aerodynamics:vertical_tail:k_ar_effective", val=np.nan)

        if self.options["low_speed_aero"]:
            self.add_input(
                "data:aerodynamics:vertical_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1"
            )
            self.add_output("data:aerodynamics:rudder:low_speed:Cy_delta_r", units="rad**-1")
        else:
            self.add_input(
                "data:aerodynamics:vertical_tail:cruise:CL_alpha", val=np.nan, units="rad**-1"
            )
            self.add_output("data:aerodynamics:rudder:cruise:Cy_delta_r", units="rad**-1")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        taper_ratio_vt = inputs["data:geometry:vertical_tail:taper_ratio"]
        aspect_ratio_vt = float(inputs["data:geometry:vertical_tail:aspect_ratio"])
        thickness_ratio_vt = float(inputs["data:geometry:vertical_tail:thickness_ratio"])
        rudder_chord_ratio = inputs["data:geometry:vertical_tail:rudder:chord_ratio"]
        k_ar_effective = float(inputs["data:aerodynamics:vertical_tail:k_ar_effective"])

        if self.options["low_speed_aero"]:
            cl_alpha_vt = inputs["data:aerodynamics:vertical_tail:low_speed:CL_alpha"]
        else:
            cl_alpha_vt = inputs["data:aerodynamics:vertical_tail:cruise:CL_alpha"]

        cl_alpha_vt_airfoil = inputs["data:aerodynamics:vertical_tail:airfoil:CL_alpha"]

        # Assumed that the rudder covers more or less all of the vertical tail while leaving a
        # small gap at the bottom and at the top
        eta_in = 0.05
        eta_out = 0.95
        kb = self.k_b_flaps(eta_in, eta_out, taper_ratio_vt)

        # Interpolation of the first graph of figure 8.53 of Roskam
        rudder_effectiveness_parameter = self.a_delta_airfoil(rudder_chord_ratio)
        k_a_delta = self.k_a_delta(
            float(rudder_effectiveness_parameter), k_ar_effective * aspect_ratio_vt
        )

        k_cl_delta = self.k_cl_delta_plain_flap(
            thickness_ratio_vt, cl_alpha_vt_airfoil, rudder_chord_ratio
        )

        cl_delta_th = self.cl_delta_theory_plain_flap(thickness_ratio_vt, rudder_chord_ratio)

        cy_delta_r = cl_alpha_vt / cl_alpha_vt_airfoil * kb * k_a_delta * k_cl_delta * cl_delta_th

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:rudder:low_speed:Cy_delta_r"] = cy_delta_r
        else:
            outputs["data:aerodynamics:rudder:cruise:Cy_delta_r"] = cy_delta_r

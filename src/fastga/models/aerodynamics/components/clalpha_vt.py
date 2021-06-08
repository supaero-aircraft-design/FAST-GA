"""
    Estimation of vertical tail lift coefficient
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
import openmdao.api as om


class ComputeClalphaVT(om.ExplicitComponent):
    """ Vertical tail lift coefficient estimation

    Based on : Roskam, Jan. Airplane Design: Part 6-Preliminary Calculation of Aerodynamic, Thrust and Power
    Characteristics. DARcorporation, 1985. Equation (8.22) applied with the geometric characteristics of the VTP and
    an effective aspect ratio different from the geometric one
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
        else:
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)

        self.add_input("data:aerodynamics:vertical_tail:airfoil:Cl_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:geometry:has_T_tail", val=np.nan)
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="deg")

        if self.options["low_speed_aero"]:
            self.add_output("data:aerodynamics:vertical_tail:low_speed:CL_alpha", units="rad**-1")
        else:
            self.add_output("data:aerodynamics:vertical_tail:cruise:CL_alpha", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        if self.options["low_speed_aero"]:
            mach = inputs["data:aerodynamics:low_speed:mach"]
            beta = math.sqrt(1 - mach ** 2)
            k = inputs["data:aerodynamics:vertical_tail:airfoil:Cl_alpha"] / (2. * np.pi)
        else:
            mach = inputs["data:aerodynamics:cruise:mach"]
            beta = math.sqrt(1 - mach ** 2)
            k = inputs["data:aerodynamics:vertical_tail:airfoil:Cl_alpha"] / (beta * 2. * np.pi)

        tail_type = np.round(inputs["data:geometry:has_T_tail"])
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        k_ar_effective = 2.9 if tail_type == 1 else 1.55
        lambda_vt = inputs["data:geometry:vertical_tail:aspect_ratio"] * k_ar_effective

        cl_alpha_vt = (
            0.8
            * 2
            * math.pi
            * lambda_vt
            / (
                2
                + math.sqrt(
                    4
                    + lambda_vt ** 2
                    * beta ** 2
                    / k ** 2
                    * (1 + (math.tan(sweep_25_vt / 180.0 * math.pi)) ** 2 / beta ** 2)
                )
            )
        )

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:vertical_tail:low_speed:CL_alpha"] = cl_alpha_vt
        else:
            outputs["data:aerodynamics:vertical_tail:cruise:CL_alpha"] = cl_alpha_vt

"""Estimation of vertical tail 3D lift coefficient."""
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

import math

import numpy as np
import scipy.interpolate as interp
import fastoad.api as oad

from .figure_digitization import FigureDigitization
from ..constants import SUBMODEL_CL_ALPHA_VT


@oad.RegisterSubmodel(
    SUBMODEL_CL_ALPHA_VT, "fastga.submodel.aerodynamics.vertical_tail.lift_curve_slope.legacy"
)
class ComputeClAlphaVT(FigureDigitization):
    """Vertical tail lift coefficient estimation

    Based on : Roskam, Jan. Airplane Design: Part 6-Preliminary Calculation of Aerodynamic,
    Thrust and Power Characteristics. DARcorporation, 1985. Equation (8.22) applied with the
    geometric characteristics of the VTP and an effective aspect ratio different from the
    geometric one obtained as  described in section 10.2.4.1.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
        else:
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)

        self.add_input(
            "data:aerodynamics:vertical_tail:airfoil:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input("data:geometry:has_T_tail", val=np.nan)
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")

        if self.options["low_speed_aero"]:
            self.add_output("data:aerodynamics:vertical_tail:low_speed:CL_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:vertical_tail:k_ar_effective")
        else:
            self.add_output("data:aerodynamics:vertical_tail:cruise:CL_alpha", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        if self.options["low_speed_aero"]:
            mach = inputs["data:aerodynamics:low_speed:mach"]
            beta = math.sqrt(1 - mach ** 2)
            k = inputs["data:aerodynamics:vertical_tail:airfoil:CL_alpha"] / (2.0 * np.pi)
        else:
            mach = inputs["data:aerodynamics:cruise:mach"]
            beta = math.sqrt(1 - mach ** 2)
            k = inputs["data:aerodynamics:vertical_tail:airfoil:CL_alpha"] / (beta * 2.0 * np.pi)

        tail_type = np.round(inputs["data:geometry:has_T_tail"])
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        span_vt = inputs["data:geometry:vertical_tail:span"]
        area_vt = inputs["data:geometry:vertical_tail:area"]
        taper_ratio_vt = inputs["data:geometry:vertical_tail:taper_ratio"]
        root_chord_vt = inputs["data:geometry:vertical_tail:root:chord"]
        area_ht = inputs["data:geometry:horizontal_tail:area"]

        l_ar = inputs["data:geometry:fuselage:rear_length"]
        w_max = inputs["data:geometry:fuselage:maximum_width"]
        h_max = inputs["data:geometry:fuselage:maximum_height"]

        avg_fus_depth = np.sqrt(w_max * h_max) * root_chord_vt / (2.0 * l_ar)

        # Compute the effect of fuselage and HTP as end plates which gives a different effective
        # aspect ratio
        k_ar_fuselage = self.k_ar_fuselage(taper_ratio_vt, span_vt, avg_fus_depth)

        k_ar_fuselage_ht = 1.7 if tail_type == 1.0 else 1.2

        k_vh = self.k_vh(float(area_ht / area_vt))

        k_ar_effective = k_ar_fuselage * (1.0 + k_vh * (k_ar_fuselage_ht - 1.0))

        lambda_vt = inputs["data:geometry:vertical_tail:aspect_ratio"] * k_ar_effective

        if span_vt / avg_fus_depth < 2.0:
            kv = 0.75
        elif span_vt / avg_fus_depth < 3.5:
            kv = interp.interp1d([2.0, 3.5], [0.75, 1.0])(float(span_vt / avg_fus_depth))
        else:
            kv = 1.0

        cl_alpha_vt = (
            kv
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
            outputs["data:aerodynamics:vertical_tail:k_ar_effective"] = k_ar_effective
        else:
            outputs["data:aerodynamics:vertical_tail:cruise:CL_alpha"] = cl_alpha_vt

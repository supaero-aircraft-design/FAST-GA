"""
    Estimation of the dependency of the aircraft lift slope coefficient as a function of Mach number
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

from ..external.xfoil import XfoilPolar
from ..constants import POLAR_POINT_COUNT, MACH_NB_PTS

from fastoad.model_base import Atmosphere


class ComputeMachInterpolation(om.Group):

    def initialize(self):
        self.options.declare('wing_airfoil_file', default="naca23012.af", types=str, allow_none=True)
        self.options.declare('htp_airfoil_file', default="naca0012.af", types=str, allow_none=True)

    # noinspection PyTypeChecker
    def setup(self):
        ivc_conditions = om.IndepVarComp()
        ivc_conditions.add_output("mach", val=0.05)
        ivc_conditions.add_output("reynolds", val=0.5e6)
        self.add_subsystem("incompressible_conditions", ivc_conditions, promotes=[])

        self.add_subsystem("wing_airfoil",
                           XfoilPolar(
                               airfoil_file=self.options["wing_airfoil_file"],
                               symmetrical=True,
                           ), promotes=[])
        self.add_subsystem("htp_airfoil",
                           XfoilPolar(
                               airfoil_file=self.options["htp_airfoil_file"],
                               symmetrical=True,
                           ), promotes=[])
        self.add_subsystem("mach_interpolation", _ComputeMachInterpolation(), promotes=["*"])

        self.connect("incompressible_conditions.mach", "wing_airfoil.xfoil:mach")
        self.connect("incompressible_conditions.reynolds", "wing_airfoil.xfoil:reynolds")
        self.connect("incompressible_conditions.mach", "htp_airfoil.xfoil:mach")
        self.connect("incompressible_conditions.reynolds", "htp_airfoil.xfoil:reynolds")

        self.connect("wing_airfoil.xfoil:alpha", "xfoil:wing:alpha")
        self.connect("wing_airfoil.xfoil:CL", "xfoil:wing:CL")
        self.connect("htp_airfoil.xfoil:alpha", "xfoil:horizontal_tail:alpha")
        self.connect("htp_airfoil.xfoil:CL", "xfoil:horizontal_tail:CL")


class _ComputeMachInterpolation(om.ExplicitComponent):
    # Based on the equation of Roskam Part VI
    """ Lift curve slope coefficient as a function of Mach number """

    def setup(self):
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:z:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")

        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")

        nans_array = np.full(POLAR_POINT_COUNT, np.nan)
        self.add_input("xfoil:wing:alpha", val=nans_array, shape=POLAR_POINT_COUNT, units="deg")
        self.add_input("xfoil:wing:CL", val=nans_array, shape=POLAR_POINT_COUNT)
        self.add_input("xfoil:horizontal_tail:alpha", val=nans_array, shape=POLAR_POINT_COUNT, units="deg")
        self.add_input("xfoil:horizontal_tail:CL", val=nans_array, shape=POLAR_POINT_COUNT)

        self.add_input("data:aerodynamics:horizontal_tail:efficiency", val=np.nan)

        self.add_output("data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector", units="rad**-1",
                        shape=MACH_NB_PTS+1)
        self.add_output("data:aerodynamics:aircraft:mach_interpolation:mach_vector", shape=MACH_NB_PTS+1)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        sweep_25_wing = float(inputs["data:geometry:wing:sweep_25"])
        aspect_ratio_wing = float(inputs["data:geometry:wing:aspect_ratio"])
        taper_ratio_wing = float(inputs["data:geometry:wing:taper_ratio"])
        area_wing = float(inputs["data:geometry:wing:area"])
        span_wing = float(inputs["data:geometry:wing:span"])

        sweep_25_htp = float(inputs["data:geometry:horizontal_tail:sweep_25"])
        aspect_ratio_htp = float(inputs["data:geometry:horizontal_tail:aspect_ratio"])
        efficiency_htp = float(inputs["data:aerodynamics:horizontal_tail:efficiency"])
        area_htp = float(inputs["data:geometry:horizontal_tail:area"])
        lp_ht = float(inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"])
        delta_z_htp = float(inputs["data:geometry:horizontal_tail:z:from_wingMAC25"])

        fuselage_width = float(inputs["data:geometry:fuselage:maximum_width"])
        fuselage_height = float(inputs["data:geometry:fuselage:maximum_height"])
        fuselage_diameter = np.sqrt(fuselage_width * fuselage_height)

        area_ratio = area_htp / area_wing

        sos_cruise = Atmosphere(inputs["data:mission:sizing:main_route:cruise:altitude"],
                                altitude_in_feet=False).speed_of_sound
        mach_cruise = float(inputs["data:TLAR:v_cruise"]) / float(sos_cruise)

        wing_cl = inputs["xfoil:wing:CL"]
        wing_alpha = inputs["xfoil:wing:alpha"]
        index_1 = np.where(wing_alpha == 1.0)
        index_2 = np.where(wing_alpha == 11.0)
        wing_airfoil_cl_alpha = (wing_cl[index_2] - wing_cl[index_1]) / (10. * math.pi / 180.)

        htp_cl = inputs["xfoil:horizontal_tail:CL"]
        htp_alpha = inputs["xfoil:horizontal_tail:alpha"]
        index_3 = np.where(htp_alpha == 1.0)
        index_4 = np.where(htp_alpha == 11.0)
        htp_airfoil_cl_alpha = (htp_cl[index_4] - htp_cl[index_3]) / (10. * math.pi / 180.)

        mach_array = np.linspace(0., 1.55 * mach_cruise, MACH_NB_PTS + 1)

        beta = np.sqrt(1. - mach_array ** 2.)
        k_wing = wing_airfoil_cl_alpha / beta / (2. * math.pi)
        tan_sweep_wing = math.tan(sweep_25_wing * math.pi / 180.)
        cos_sweep_wing = math.cos(sweep_25_wing * math.pi / 180.)

        wing_cl_alpha = (2. * math.pi * aspect_ratio_wing) / (
                2. + np.sqrt(
                        aspect_ratio_wing ** 2. * beta ** 2. / k_wing ** 2. * (
                            1. + tan_sweep_wing ** 2. / beta ** 2.
                        ) + 4.
                        )
        )

        k_htp = htp_airfoil_cl_alpha / beta / (2. * math.pi)
        tan_sweep_htp = math.tan(sweep_25_htp * math.pi / 180.)

        # Computing the fuselage interference factor
        k_wf = 1 + 0.025 * (fuselage_diameter / span_wing) - 0.25 * (fuselage_diameter / span_wing) ** 2.0

        htp_cl_alpha = (2. * math.pi * aspect_ratio_htp) / (
                2. + np.sqrt(
                        aspect_ratio_htp ** 2. * beta ** 2. / k_htp ** 2. * (
                            1. + tan_sweep_htp ** 2. / beta ** 2.
                        ) + 4.
                        )
        )

        k_a = 1. / aspect_ratio_wing - 1. / (1. + aspect_ratio_wing ** 1.7)
        k_lambda = (10. - 3. * taper_ratio_wing) / 7.
        k_h = (1. - delta_z_htp / span_wing) / (2. * lp_ht / span_wing) ** (1. / 3.)

        downwash_gradient = 4.44 * (
                                k_a * k_lambda * k_h * np.sqrt(cos_sweep_wing)
                             ) ** 1.19 * wing_cl_alpha / wing_cl_alpha[0]

        aircraft_cl_alpha = k_wf * wing_cl_alpha + htp_cl_alpha * efficiency_htp * area_ratio * (1. - downwash_gradient)

        outputs["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"] = aircraft_cl_alpha
        outputs["data:aerodynamics:aircraft:mach_interpolation:mach_vector"] = mach_array

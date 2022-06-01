"""Estimation of fuselage weight."""
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
import openmdao.api as om
import math

import fastoad.api as oad
from stdatm import Atmosphere

from .constants import SUBMODEL_FUSELAGE_MASS

oad.RegisterSubmodel.active_models[
    SUBMODEL_FUSELAGE_MASS
] = "fastga.submodel.weight.mass.airframe.fuselage.legacy"


@oad.RegisterSubmodel(
    SUBMODEL_FUSELAGE_MASS, "fastga.submodel.weight.mass.airframe.fuselage.legacy"
)
class ComputeFuselageWeight(om.ExplicitComponent):
    """
    Fuselage weight estimation

    Based on a statistical analysis. See :cite:`nicolai:2010` but can also be found in
    :cite:`gudmundsson:2013`.
    """

    def setup(self):
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:fuselage:k_factor", val=1.0)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:TLAR:v_max_sl", val=np.nan, units="kn")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")

        self.add_output("data:weight:airframe:fuselage:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        maximum_width = inputs["data:geometry:fuselage:maximum_width"]
        maximum_height = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]
        v_max_sl = inputs["data:TLAR:v_max_sl"]

        a2 = (
            200.0
            * (
                (mtow * sizing_factor_ultimate / (10.0 ** 5.0)) ** 0.286
                * (fus_length * 3.28084 / 10.0) ** 0.857
                * (maximum_width + maximum_height)
                * 3.28084
                / 10.0
                * (v_max_sl / 100.0) ** 0.338
            )
            ** 1.1
        )  # mass formula in lb

        outputs["data:weight:airframe:fuselage:mass"] = (
            a2 * inputs["data:weight:airframe:fuselage:k_factor"]
        )


@oad.RegisterSubmodel(
    SUBMODEL_FUSELAGE_MASS, "fastga.submodel.weight.mass.airframe.fuselage.raymer"
)
class ComputeFuselageWeightRaymer(om.ExplicitComponent):
    """
    Fuselage weight estimation

    Based on : Raymer, Daniel. Aircraft design: a conceptual approach. American Institute of
    Aeronautics and Astronautics, Inc., 2012.

    Can also be found in : Gudmundsson, Snorri. General aviation aircraft design: Applied Methods
    and Procedures. Butterworth-Heinemann, 2013. Equation (6-25).
    """

    def setup(self):

        self.add_input("data:geometry:fuselage:length", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:wet_area", val=np.nan, units="ft**2")
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:fuselage:k_factor", val=1.0)
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="ft"
        )
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="kn")

        self.add_output("data:weight:airframe:fuselage:mass_raymer", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]
        maximum_width = inputs["data:geometry:fuselage:maximum_width"]
        maximum_height = inputs["data:geometry:fuselage:maximum_height"]
        wet_area_fus = inputs["data:geometry:fuselage:wet_area"]
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]
        v_cruise = inputs["data:TLAR:v_cruise"] * 0.5144

        atm_cruise = Atmosphere(cruise_alt)
        rho_cruise = atm_cruise.density
        pressure_cruise = atm_cruise.pressure

        atm_sl = Atmosphere(0.0)
        pressure_sl = atm_sl.pressure

        dynamic_pressure = 1.0 / 2.0 * rho_cruise * v_cruise ** 2.0 * 0.020885434273039

        if cruise_alt > 10000.0:
            fus_dia = (maximum_height + maximum_width) / 2.0
            v_press = (fus_length - lar - lav) * math.pi * (fus_dia / 2.0) ** 2.0
            delta_p = (pressure_sl - pressure_cruise) * 0.000145038
        else:
            v_press = 0.0
            delta_p = 0.0

        a2 = 0.052 * (
            wet_area_fus ** 1.086
            * (sizing_factor_ultimate * mtow) ** 0.177
            * lp_ht ** (-0.051)
            * ((fus_length - lar - lav) / maximum_height) ** (-0.072)
            * dynamic_pressure ** 0.241
            + 11.9 * (v_press * delta_p) ** 0.271
        )

        outputs["data:weight:airframe:fuselage:mass_raymer"] = (
            a2 * inputs["data:weight:airframe:fuselage:k_factor"]
        )

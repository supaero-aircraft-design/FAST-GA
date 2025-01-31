"""
Python module for fuselage weight calculation, part of the airframe mass computation.
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

import logging

import fastoad.api as oad
import numpy as np
import openmdao.api as om
from stdatm import AtmosphereWithPartials

from .constants import (
    SERVICE_FUSELAGE_MASS,
    SUBMODEL_FUSELAGE_MASS_LEGACY,
    SUBMODEL_FUSELAGE_MASS_RAYMER,
    SUBMODEL_FUSELAGE_MASS_ROSKAM,
)

_LOGGER = logging.getLogger(__name__)

oad.RegisterSubmodel.active_models[SERVICE_FUSELAGE_MASS] = SUBMODEL_FUSELAGE_MASS_LEGACY


@oad.RegisterSubmodel(SERVICE_FUSELAGE_MASS, SUBMODEL_FUSELAGE_MASS_LEGACY)
class ComputeFuselageWeight(om.ExplicitComponent):
    """
    Fuselage weight estimation

    Based on a statistical analysis. See :cite:`nicolai:2010` but can also be found in
    :cite:`gudmundsson:2013`.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:fuselage:k_factor", val=1.0)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:TLAR:v_max_sl", val=np.nan, units="kn")

        self.add_output("data:weight:airframe:fuselage:mass", units="lb")

        self.declare_partials("*", "*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
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
                (mtow * sizing_factor_ultimate / (10.0**5.0)) ** 0.286
                * (fus_length * 3.28084 / 10.0) ** 0.857
                * (maximum_width + maximum_height)
                * 0.328084
                * (v_max_sl / 100.0) ** 0.338
            )
            ** 1.1
        )  # mass formula in lb

        outputs["data:weight:airframe:fuselage:mass"] = (
            a2 * inputs["data:weight:airframe:fuselage:k_factor"]
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        maximum_width = inputs["data:geometry:fuselage:maximum_width"]
        maximum_height = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]
        v_max_sl = inputs["data:TLAR:v_max_sl"]
        k_factor = inputs["data:weight:airframe:fuselage:k_factor"]

        partials[
            "data:weight:airframe:fuselage:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor * (
            200.0
            * 0.3146
            * (
                (mtow / (10.0**5.0)) ** 0.286
                * (fus_length * 3.28084 / 10.0) ** 0.857
                * (maximum_width + maximum_height)
                * 0.328084
                * (v_max_sl / 100.0) ** 0.338
            )
            ** 1.1
            * sizing_factor_ultimate**-0.6854
        )
        partials["data:weight:airframe:fuselage:mass", "data:weight:aircraft:MTOW"] = k_factor * (
            200.0
            * 0.3146
            * (
                (sizing_factor_ultimate / (10.0**5.0)) ** 0.286
                * (fus_length * 3.28084 / 10.0) ** 0.857
                * (maximum_width + maximum_height)
                * 0.328084
                * (v_max_sl / 100.0) ** 0.338
            )
            ** 1.1
            * mtow**-0.6854
        )
        partials["data:weight:airframe:fuselage:mass", "data:geometry:fuselage:maximum_width"] = (
            k_factor
            * (
                200.0
                * 1.1
                * (
                    (mtow * sizing_factor_ultimate / (10.0**5.0)) ** 0.286
                    * (fus_length * 3.28084 / 10.0) ** 0.857
                    * 0.328084
                    * (v_max_sl / 100.0) ** 0.338
                )
                ** 1.1
                * (maximum_width + maximum_height) ** 0.1
            )
        )
        partials["data:weight:airframe:fuselage:mass", "data:geometry:fuselage:maximum_height"] = (
            k_factor
            * (
                200.0
                * 1.1
                * (
                    (mtow * sizing_factor_ultimate / (10.0**5.0)) ** 0.286
                    * (fus_length * 3.28084 / 10.0) ** 0.857
                    * 0.328084
                    * (v_max_sl / 100.0) ** 0.338
                )
                ** 1.1
                * (maximum_width + maximum_height) ** 0.1
            )
        )
        partials["data:weight:airframe:fuselage:mass", "data:geometry:fuselage:length"] = (
            k_factor
            * 200.0
            * 0.9427
            * (
                (mtow * sizing_factor_ultimate / (10.0**5.0)) ** 0.286
                * 0.328084**0.857
                * (maximum_width + maximum_height)
                * 0.328084
                * (v_max_sl / 100.0) ** 0.338
            )
            ** 1.1
            * fus_length ** (-0.0573)
        )
        partials["data:weight:airframe:fuselage:mass", "data:TLAR:v_max_sl"] = k_factor * (
            200.0
            * 0.3718
            * (
                (mtow * sizing_factor_ultimate / (10.0**5.0)) ** 0.286
                * (fus_length * 3.28084 / 10.0) ** 0.857
                * (maximum_width + maximum_height)
                * 0.328084
                * 0.01**0.338
            )
            ** 1.1
            * v_max_sl**-0.6282
        )
        partials["data:weight:airframe:fuselage:mass", "data:weight:airframe:fuselage:k_factor"] = (
            200.0
            * (
                (mtow * sizing_factor_ultimate / (10.0**5.0)) ** 0.286
                * (fus_length * 3.28084 / 10.0) ** 0.857
                * (maximum_width + maximum_height)
                * 0.328084
                * (v_max_sl / 100.0) ** 0.338
            )
            ** 1.1
        )


@oad.RegisterSubmodel(SERVICE_FUSELAGE_MASS, SUBMODEL_FUSELAGE_MASS_RAYMER)
class ComputeFuselageWeightRaymer(om.ExplicitComponent):
    """
    Fuselage weight estimation

    Based on : Raymer, Daniel. Aircraft design: a conceptual approach. American Institute of
    Aeronautics and Astronautics, Inc., 2012.

    Can also be found in : Gudmundsson, Snorri. General aviation aircraft design: Applied Methods
    and Procedures. Butterworth-Heinemann, 2013. Equation (6-25).
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
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
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")

        self.add_output("data:weight:airframe:fuselage:mass", units="lb")

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
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
        v_cruise = inputs["data:TLAR:v_cruise"]

        atm_cruise = AtmosphereWithPartials(cruise_alt)
        rho_cruise = atm_cruise.density
        pressure_cruise = atm_cruise.pressure
        atm_sl = AtmosphereWithPartials(0.0)
        pressure_sl = atm_sl.pressure

        dynamic_pressure = 1.0 / 2.0 * rho_cruise * v_cruise**2.0 * 0.020885434273039

        if cruise_alt > 10000.0:
            is_pressurized = 1.0
        else:
            is_pressurized = 0.0

        # is_pressurized is an option that affects the fuselage sizing.
        # It describes whether the fuselage is pressurized or not depending on the cruise altitude.

        fus_dia = (maximum_height + maximum_width) / 2.0
        v_press = (fus_length - lar - lav) * np.pi * (fus_dia / 2.0) ** 2.0
        delta_p = (pressure_sl - pressure_cruise) * 0.000145038

        a2 = 0.052 * (
            wet_area_fus**1.086
            * (sizing_factor_ultimate * mtow) ** 0.177
            * lp_ht ** (-0.051)
            * ((fus_length - lar - lav) / maximum_height) ** (-0.072)
            * dynamic_pressure**0.241
            + 11.9 * (v_press * delta_p) ** 0.271 * is_pressurized
        )

        outputs["data:weight:airframe:fuselage:mass"] = (
            a2 * inputs["data:weight:airframe:fuselage:k_factor"]
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
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
        v_cruise = inputs["data:TLAR:v_cruise"]

        k_factor = inputs["data:weight:airframe:fuselage:k_factor"]

        atm_cruise = AtmosphereWithPartials(cruise_alt)
        rho_cruise = atm_cruise.density
        pressure_cruise = atm_cruise.pressure
        atm_sl = AtmosphereWithPartials(0.0)
        pressure_sl = atm_sl.pressure

        dynamic_pressure = 1.0 / 2.0 * rho_cruise * v_cruise**2.0 * 0.020885434273039
        fus_dia = (maximum_height + maximum_width) / 2.0
        v_press = (fus_length - lar - lav) * np.pi * (fus_dia / 2.0) ** 2.0
        delta_p = (pressure_sl - pressure_cruise) * 0.000145038

        if cruise_alt > 10000.0:
            is_pressurized = 1.0
        else:
            is_pressurized = 0.0

        # is_pressurized is an option that affects the fuselage sizing.
        # It describes whether the fuselage is pressurized or not depending on the cruise altitude.

        partials["data:weight:airframe:fuselage:mass", "data:weight:airframe:fuselage:k_factor"] = (
            0.052
            * (
                wet_area_fus**1.086
                * (sizing_factor_ultimate * mtow) ** 0.177
                * lp_ht ** (-0.051)
                * ((fus_length - lar - lav) / maximum_height) ** (-0.072)
                * dynamic_pressure**0.241
                + 11.9 * (v_press * delta_p) ** 0.271 * is_pressurized
            )
        )
        partials["data:weight:airframe:fuselage:mass", "data:geometry:fuselage:length"] = (
            k_factor
            * (
                0.052
                * (
                    wet_area_fus**1.086
                    * (sizing_factor_ultimate * mtow) ** 0.177
                    * lp_ht ** (-0.051)
                    * -0.072
                    * (fus_length - lar - lav) ** (-1.072)
                    * maximum_height**0.072
                    * dynamic_pressure**0.241
                    + 11.9
                    * is_pressurized
                    * 0.271
                    * (fus_length - lar - lav) ** -0.729
                    * (delta_p * np.pi * (fus_dia / 2.0) ** 2.0) ** 0.271
                )
            )
        )
        partials["data:weight:airframe:fuselage:mass", "data:geometry:fuselage:front_length"] = (
            k_factor
            * (
                0.052
                * (
                    wet_area_fus**1.086
                    * (sizing_factor_ultimate * mtow) ** 0.177
                    * lp_ht ** (-0.051)
                    * maximum_height**0.072
                    * 0.072
                    * (fus_length - lar - lav) ** -1.072
                    * dynamic_pressure**0.241
                    - 11.9
                    * is_pressurized
                    * 0.271
                    * (fus_length - lar - lav) ** (-0.729)
                    * (delta_p * np.pi * (fus_dia / 2.0) ** 2.0) ** 0.271
                )
            )
        )

        partials["data:weight:airframe:fuselage:mass", "data:geometry:fuselage:rear_length"] = (
            k_factor
            * (
                0.052
                * (
                    wet_area_fus**1.086
                    * (sizing_factor_ultimate * mtow) ** 0.177
                    * lp_ht ** (-0.051)
                    * maximum_height**0.072
                    * 0.072
                    * (fus_length - lar - lav) ** -1.072
                    * dynamic_pressure**0.241
                    - 11.9
                    * is_pressurized
                    * 0.271
                    * (fus_length - lar - lav) ** (-0.729)
                    * (delta_p * np.pi * (fus_dia / 2.0) ** 2.0) ** 0.271
                )
            )
        )

        partials["data:weight:airframe:fuselage:mass", "data:geometry:fuselage:maximum_width"] = (
            k_factor
            * 0.052
            * is_pressurized
            * 11.9
            * 0.271
            * (delta_p * np.pi * (fus_length - lar - lav)) ** 0.271
            * (maximum_height + maximum_width) ** (-0.458)
            * 2 ** (-0.084)
        )

        partials["data:weight:airframe:fuselage:mass", "data:geometry:fuselage:maximum_height"] = (
            k_factor
            * (
                0.052
                * (
                    wet_area_fus**1.086
                    * (sizing_factor_ultimate * mtow) ** 0.177
                    * lp_ht ** (-0.051)
                    * (fus_length - lar - lav) ** (-0.072)
                    * dynamic_pressure**0.241
                    * 0.072
                    * maximum_height**-0.928
                    + is_pressurized
                    * 11.9
                    * 0.271
                    * (delta_p * np.pi * (fus_length - lar - lav)) ** 0.271
                    * (maximum_height + maximum_width) ** (-0.458)
                    * 2 ** (-0.084)
                )
            )
        )

        partials["data:weight:airframe:fuselage:mass", "data:geometry:fuselage:wet_area"] = (
            k_factor
            * 0.052
            * (
                1.086
                * wet_area_fus**0.086
                * (sizing_factor_ultimate * mtow) ** 0.177
                * lp_ht ** (-0.051)
                * ((fus_length - lar - lav) / maximum_height) ** (-0.072)
                * dynamic_pressure**0.241
            )
        )
        partials[
            "data:weight:airframe:fuselage:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor * (
            0.052
            * (
                wet_area_fus**1.086
                * 0.177
                * (sizing_factor_ultimate * mtow) ** -0.823
                * mtow
                * lp_ht ** (-0.051)
                * ((fus_length - lar - lav) / maximum_height) ** (-0.072)
                * dynamic_pressure**0.241
            )
        )
        partials["data:weight:airframe:fuselage:mass", "data:weight:aircraft:MTOW"] = k_factor * (
            0.052
            * (
                wet_area_fus**1.086
                * 0.177
                * (sizing_factor_ultimate * mtow) ** -0.823
                * sizing_factor_ultimate
                * lp_ht ** (-0.051)
                * ((fus_length - lar - lav) / maximum_height) ** (-0.072)
                * dynamic_pressure**0.241
            )
        )
        partials[
            "data:weight:airframe:fuselage:mass",
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
        ] = k_factor * (
            -0.052
            * (
                wet_area_fus**1.086
                * (sizing_factor_ultimate * mtow) ** 0.177
                * 0.051
                * lp_ht ** (-1.051)
                * ((fus_length - lar - lav) / maximum_height) ** (-0.072)
                * dynamic_pressure**0.241
            )
        )

        partials["data:weight:airframe:fuselage:mass", "data:TLAR:v_cruise"] = (
            k_factor
            * 0.052
            * wet_area_fus**1.086
            * (sizing_factor_ultimate * mtow) ** 0.177
            * lp_ht ** (-0.051)
            * ((fus_length - lar - lav) / maximum_height) ** (-0.072)
            * 2
            * 0.241
            * (0.5 * rho_cruise * 0.020885434273039) ** 0.241
            * v_cruise ** (-0.518)
        )

        partials[
            "data:weight:airframe:fuselage:mass", "data:mission:sizing:main_route:cruise:altitude"
        ] = k_factor * (
            0.052
            * (
                wet_area_fus**1.086
                * (sizing_factor_ultimate * mtow) ** 0.177
                * lp_ht ** (-0.051)
                * ((fus_length - lar - lav) / maximum_height) ** (-0.072)
                * 0.241
                * (0.5 * v_cruise**2.0 * 0.020885434273039) ** 0.241
                * atm_cruise.partial_density_altitude
                * rho_cruise ** (-0.759)
                - 11.9
                * is_pressurized
                * 0.271
                * (v_press * 0.000145038) ** 0.271
                * atm_cruise.partial_pressure_altitude
                * (pressure_sl - pressure_cruise) ** (-0.729)
            )
        )


@oad.RegisterSubmodel(SERVICE_FUSELAGE_MASS, SUBMODEL_FUSELAGE_MASS_ROSKAM)
class ComputeFuselageWeightRoskam(om.ExplicitComponent):
    """
    Fuselage weight estimation, includes the computation of the fuselage weight for a high wing
    aircraft.

    Based on : Roskam. Airplane design - Part 5: component weight estimation

    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="ft")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="ft")
        self.add_input("data:geometry:wing_configuration", val=np.nan)

        self.add_output("data:weight:airframe:fuselage:mass", units="lb")

        self.declare_partials(
            of="*",
            wrt=[
                "data:geometry:fuselage:length",
                "data:geometry:fuselage:front_length",
                "data:weight:aircraft:MTOW",
                "data:geometry:cabin:seats:passenger:NPAX_max",
                "data:geometry:fuselage:maximum_width",
                "data:geometry:fuselage:maximum_width",
                "data:geometry:fuselage:maximum_height",
            ],
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        maximum_width = inputs["data:geometry:fuselage:maximum_width"]
        maximum_height = inputs["data:geometry:fuselage:maximum_height"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        npax_max = (
            inputs["data:geometry:cabin:seats:passenger:NPAX_max"] + 2.0
        )  # addition of 2 pilots
        wing_config = inputs["data:geometry:wing_configuration"]

        fus_dia = (maximum_height + maximum_width) / 2.0
        p_max = 2 * np.pi * (fus_dia / 2)  # maximum perimeter of the fuselage

        if wing_config == 1.0:
            # The formula found in Roskam originally contains a division by 100, but it leads to
            # results way too low. It will be omitted here. It does not seem to cause an issue
            # for the high wing configuration however, so we will simply issue a warning with a
            # recommendation to switch method for low wing aircraft
            a2 = 0.04682 * (mtow**0.692 * npax_max**0.374 * (fus_length - lav) ** 0.590)

            _LOGGER.warning(
                "This submodel is not trusted for the computation of the fuselage weight of low "
                "wing aircraft as it gives very small results. Consider switching submodel"
            )

        elif wing_config == 3.0:
            a2 = 14.86 * (
                mtow**0.144
                * ((fus_length - lav) / p_max) ** 0.778
                * (fus_length - lav) ** 0.383
                * npax_max**0.455
            )

        else:
            _LOGGER.info(
                "No formula available for the weight of the fuselage with a mid-wing "
                "configuration, taking high wing instead"
            )
            a2 = 14.86 * (
                mtow**0.144
                * ((fus_length - lav) / p_max) ** 0.778
                * (fus_length - lav) ** 0.383
                * npax_max**0.455
            )

        outputs["data:weight:airframe:fuselage:mass"] = a2

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        wing_config = inputs["data:geometry:wing_configuration"]

        if wing_config == 1.0:
            fus_length = inputs["data:geometry:fuselage:length"]
            lav = inputs["data:geometry:fuselage:front_length"]
            mtow = inputs["data:weight:aircraft:MTOW"]
            npax_max = (
                inputs["data:geometry:cabin:seats:passenger:NPAX_max"] + 2.0
            )  # addition of 2 pilots

            partials["data:weight:airframe:fuselage:mass", "data:weight:aircraft:MTOW"] = (
                0.04682 * (0.692 * mtow**-0.308 * npax_max**0.374 * (fus_length - lav) ** 0.590)
            )
            partials[
                "data:weight:airframe:fuselage:mass", "data:geometry:cabin:seats:passenger:NPAX_max"
            ] = 0.04682 * (mtow**0.692 * 0.374 * npax_max**-0.626 * (fus_length - lav) ** 0.590)
            partials["data:weight:airframe:fuselage:mass", "data:geometry:fuselage:length"] = (
                0.04682 * (mtow**0.692 * npax_max**0.374 * 0.590 * (fus_length - lav) ** -0.41)
            )
            partials[
                "data:weight:airframe:fuselage:mass", "data:geometry:fuselage:front_length"
            ] = -(0.04682 * (mtow**0.692 * npax_max**0.374 * 0.590 * (fus_length - lav) ** -0.41))
            partials[
                "data:weight:airframe:fuselage:mass", "data:geometry:fuselage:maximum_width"
            ] = 0.0
            partials[
                "data:weight:airframe:fuselage:mass", "data:geometry:fuselage:maximum_height"
            ] = 0.0

        else:
            fus_length = inputs["data:geometry:fuselage:length"]
            lav = inputs["data:geometry:fuselage:front_length"]
            mtow = inputs["data:weight:aircraft:MTOW"]
            npax_max = (
                inputs["data:geometry:cabin:seats:passenger:NPAX_max"] + 2.0
            )  # addition of 2 pilots

            maximum_width = inputs["data:geometry:fuselage:maximum_width"]
            maximum_height = inputs["data:geometry:fuselage:maximum_height"]

            p_max = (
                np.pi * (maximum_height + maximum_width) / 2
            )  # maximum perimeter of the fuselage

            partials["data:weight:airframe:fuselage:mass", "data:weight:aircraft:MTOW"] = 14.86 * (
                0.144
                * mtow**-0.856
                * ((fus_length - lav) / p_max) ** 0.778
                * (fus_length - lav) ** 0.383
                * npax_max**0.455
            )
            partials[
                "data:weight:airframe:fuselage:mass", "data:geometry:cabin:seats:passenger:NPAX_max"
            ] = 14.86 * (
                mtow**0.144
                * ((fus_length - lav) / p_max) ** 0.778
                * (fus_length - lav) ** 0.383
                * 0.455
                * npax_max**-0.545
            )
            partials["data:weight:airframe:fuselage:mass", "data:geometry:fuselage:length"] = (
                14.86
                * (
                    mtow**0.144
                    * p_max**-0.778
                    * 1.161
                    * (fus_length - lav) ** 0.161
                    * npax_max**0.455
                )
            )
            partials[
                "data:weight:airframe:fuselage:mass", "data:geometry:fuselage:front_length"
            ] = -14.86 * (
                mtow**0.144 * p_max**-0.778 * 1.161 * (fus_length - lav) ** 0.161 * npax_max**0.455
            )
            partials[
                "data:weight:airframe:fuselage:mass", "data:geometry:fuselage:maximum_width"
            ] = (
                14.86
                * (
                    mtow**0.144
                    * -0.778
                    * p_max**-1.778
                    * (fus_length - lav) ** 1.161
                    * npax_max**0.455
                )
                * np.pi
                / 2.0
            )
            partials[
                "data:weight:airframe:fuselage:mass", "data:geometry:fuselage:maximum_height"
            ] = (
                14.86
                * (
                    mtow**0.144
                    * -0.778
                    * p_max**-1.778
                    * (fus_length - lav) ** 1.161
                    * npax_max**0.455
                )
                * np.pi
                / 2.0
            )

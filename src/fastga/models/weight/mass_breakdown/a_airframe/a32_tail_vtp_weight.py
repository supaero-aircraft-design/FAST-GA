"""
Python module for vtp weight calculation, part of the tail mass computation.
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

import fastoad.api as oad
import numpy as np
import openmdao.api as om
from stdatm import AtmosphereWithPartials

from .constants import SERVICE_VTP_MASS, SUBMODEL_VTP_MASS_LEGACY, SUBMODEL_VTP_MASS_GD


@oad.RegisterSubmodel(SERVICE_VTP_MASS, SUBMODEL_VTP_MASS_LEGACY)
class ComputeVTPWeight(om.ExplicitComponent):
    """
    Weight estimation for vtp weight

    Based on a statistical analysis. See :cite:`raymer:2012` but can also be found in
    :cite:`gudmundsson:2013`
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:vertical_tail:k_factor", val=1.0)
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")
        self.add_input("data:geometry:has_T_tail", val=np.nan)
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:vertical_tail:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)

        self.add_output("data:weight:airframe:vertical_tail:mass", units="lb")

        self.declare_partials(
            of="data:weight:airframe:vertical_tail:mass",
            wrt=[
                "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
                "data:weight:aircraft:MTOW",
                "data:weight:airframe:vertical_tail:k_factor",
                "data:TLAR:v_cruise",
                "data:geometry:has_T_tail",
                "data:geometry:vertical_tail:area",
                "data:geometry:vertical_tail:thickness_ratio",
                "data:geometry:vertical_tail:sweep_25",
                "data:geometry:vertical_tail:aspect_ratio",
                "data:geometry:vertical_tail:taper_ratio",
                "data:mission:sizing:main_route:cruise:altitude",
            ],
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument, too-many-local-variables
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        v_cruise_ktas = inputs["data:TLAR:v_cruise"]
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]

        rho_cruise = AtmosphereWithPartials(cruise_alt, altitude_in_feet=True).density
        dynamic_pressure = 1.0 / 2.0 * rho_cruise * v_cruise_ktas**2.0 * 0.0208854

        has_t_tail = inputs["data:geometry:has_T_tail"]
        area_vt = inputs["data:geometry:vertical_tail:area"]
        t_c_vt = inputs["data:geometry:vertical_tail:thickness_ratio"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]

        a32 = (
            0.073
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** 0.376
                * dynamic_pressure**0.122
                * area_vt**0.873
                * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                * taper_vt**0.039
            )
        )
        # Mass formula in lb

        outputs["data:weight:airframe:vertical_tail:mass"] = (
            a32 * inputs["data:weight:airframe:vertical_tail:k_factor"]
        )

    # pylint: disable=missing-function-docstring, unused-argument, too-many-local-variables
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        v_cruise_ktas = inputs["data:TLAR:v_cruise"]
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]
        atm_cruise = AtmosphereWithPartials(cruise_alt)

        k_factor_vt = inputs["data:weight:airframe:vertical_tail:k_factor"]

        has_t_tail = inputs["data:geometry:has_T_tail"]
        area_vt = inputs["data:geometry:vertical_tail:area"]
        t_c_vt = inputs["data:geometry:vertical_tail:thickness_ratio"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]

        rho_cruise = atm_cruise.density
        dynamic_pressure = 1.0 / 2.0 * rho_cruise * v_cruise_ktas**2.0 * 0.0208854

        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor_vt * (
            0.073
            * 0.376
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** (-0.624)
                * mtow
                * dynamic_pressure**0.122
                * area_vt**0.873
                * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                * taper_vt**0.039
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:weight:aircraft:MTOW"] = (
            k_factor_vt
            * (
                0.073
                * 0.376
                * (1.0 + 0.2 * has_t_tail)
                * (
                    (sizing_factor_ultimate * mtow) ** (-0.624)
                    * sizing_factor_ultimate
                    * dynamic_pressure**0.122
                    * area_vt**0.873
                    * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                    * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                    * taper_vt**0.039
                )
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:TLAR:v_cruise"] = k_factor_vt * (
            0.073
            * 0.244
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** 0.376
                * (0.5 * rho_cruise * 0.0208854) ** 0.122
                * v_cruise_ktas ** (-0.756)
                * area_vt**0.873
                * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                * taper_vt**0.039
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:geometry:has_T_tail"] = (
            k_factor_vt
            * (
                0.073
                * 0.2
                * (
                    (sizing_factor_ultimate * mtow) ** 0.376
                    * dynamic_pressure**0.122
                    * area_vt**0.873
                    * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                    * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                    * taper_vt**0.039
                )
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:area"] = (
            k_factor_vt
            * (
                0.073
                * 0.873
                * (1.0 + 0.2 * has_t_tail)
                * (
                    (sizing_factor_ultimate * mtow) ** 0.376
                    * dynamic_pressure**0.122
                    * area_vt ** (-0.127)
                    * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                    * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                    * taper_vt**0.039
                )
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:thickness_ratio"
        ] = k_factor_vt * (
            -0.073
            * 0.49
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** 0.376
                * dynamic_pressure**0.122
                * area_vt**0.873
                * (100.0 / np.cos(sweep_25_vt)) ** -0.49
                * t_c_vt ** (-1.49)
                * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                * taper_vt**0.039
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:sweep_25"
        ] = k_factor_vt * (
            0.073
            * 0.224
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** 0.376
                * dynamic_pressure**0.122
                * area_vt**0.873
                * (100.0 * t_c_vt) ** -0.49
                * ar_vt**0.357
                * np.tan(sweep_25_vt)
                * np.cos(sweep_25_vt) ** (-0.224)
                * taper_vt**0.039
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:aspect_ratio"
        ] = k_factor_vt * (
            0.073
            * 0.357
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** 0.376
                * dynamic_pressure**0.122
                * area_vt**0.873
                * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                * (1 / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                * ar_vt ** (-0.643)
                * taper_vt**0.039
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:taper_ratio"
        ] = k_factor_vt * (
            0.073
            * 0.039
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** 0.376
                * dynamic_pressure**0.122
                * area_vt**0.873
                * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                * taper_vt ** (-0.961)
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:weight:airframe:vertical_tail:k_factor"
        ] = (
            0.073
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** 0.376
                * dynamic_pressure**0.122
                * area_vt**0.873
                * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                * taper_vt**0.039
            )
        )

        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:mission:sizing:main_route:cruise:altitude",
        ] = k_factor_vt * (
            0.073
            * 0.122
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** 0.376
                * (0.5 * v_cruise_ktas**2.0 * 0.0208854) ** 0.122
                * rho_cruise ** (-0.878)
                * atm_cruise.partial_density_altitude
                * area_vt**0.873
                * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                * taper_vt**0.039
            )
        )


@oad.RegisterSubmodel(SERVICE_VTP_MASS, SUBMODEL_VTP_MASS_GD)
class ComputeVTPWeightGD(om.ExplicitComponent):
    """
    Weight estimation for vtp weight

    Based on a statistical analysis. See :cite:`roskampart5:1985` traditionally used on
    commercial aircraft but found to work fairly well on high performance GA aircraft.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:TLAR:v_max_sl", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:vertical_tail:k_factor", val=1.0)

        self.add_input("data:geometry:has_T_tail", val=np.nan)
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="ft**2")
        self.add_input(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="ft"
        )
        # Rudder is assumed to take full span so rudder ratio is equal to chord ratio
        self.add_input("data:geometry:vertical_tail:rudder:chord_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)

        self.add_output("data:weight:airframe:vertical_tail:mass", units="lb")

        self.declare_partials(
            of="data:weight:airframe:vertical_tail:mass",
            wrt=[
                "data:geometry:has_T_tail",
                "data:weight:aircraft:MTOW",
                "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
                "data:TLAR:v_max_sl",
                "data:geometry:vertical_tail:area",
                "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:vertical_tail:rudder:chord_ratio",
                "data:geometry:vertical_tail:aspect_ratio",
                "data:geometry:vertical_tail:taper_ratio",
                "data:geometry:vertical_tail:sweep_25",
                "data:weight:airframe:vertical_tail:k_factor",
            ],
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument, too-many-local-variables
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]

        has_t_tail = inputs["data:geometry:has_T_tail"]
        area_vt = inputs["data:geometry:vertical_tail:area"]
        v_h = inputs["data:TLAR:v_max_sl"]
        atm0 = AtmosphereWithPartials(0)
        atm0.true_airspeed = v_h
        mach_h = atm0.mach
        lp_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        rudder_chord_ratio = inputs["data:geometry:vertical_tail:rudder:chord_ratio"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]

        a32 = (
            0.19
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
        )
        # Mass formula in lb

        outputs["data:weight:airframe:vertical_tail:mass"] = (
            a32 * inputs["data:weight:airframe:vertical_tail:k_factor"]
        )

    # pylint: disable=missing-function-docstring, unused-argument, too-many-local-variables
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]

        has_t_tail = inputs["data:geometry:has_T_tail"]
        area_vt = inputs["data:geometry:vertical_tail:area"]
        lp_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        rudder_chord_ratio = inputs["data:geometry:vertical_tail:rudder:chord_ratio"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]
        v_h = inputs["data:TLAR:v_max_sl"]

        k_factor_vt = inputs["data:weight:airframe:vertical_tail:k_factor"]

        atm0 = AtmosphereWithPartials(0)
        atm0.true_airspeed = v_h
        mach_h = atm0.mach

        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor_vt * (
            0.19
            * 1.014
            * 0.363
            * (
                (1 + has_t_tail) ** 0.5
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
            * (sizing_factor_ultimate * mtow) ** (-0.631918)
            * mtow
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:weight:aircraft:MTOW"] = (
            k_factor_vt
            * (
                0.19
                * 1.014
                * 0.363
                * (
                    (1 + has_t_tail) ** 0.5
                    * area_vt**1.089
                    * mach_h**0.601
                    * lp_vt**-0.726
                    * (1 + rudder_chord_ratio) ** 0.217
                    * ar_vt**0.337
                    * (1 + taper_vt) ** 0.363
                    * np.cos(sweep_25_vt) ** -0.484
                )
                ** 1.014
                * (sizing_factor_ultimate * mtow) ** (-0.631918)
                * sizing_factor_ultimate
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:geometry:has_T_tail"] = (
            k_factor_vt
            * (
                0.19
                * 1.014
                * 0.5
                * (
                    (sizing_factor_ultimate * mtow) ** 0.363
                    * area_vt**1.089
                    * mach_h**0.601
                    * lp_vt**-0.726
                    * (1 + rudder_chord_ratio) ** 0.217
                    * ar_vt**0.337
                    * (1 + taper_vt) ** 0.363
                    * np.cos(sweep_25_vt) ** -0.484
                )
                ** 1.014
            )
            * (1 + has_t_tail) ** -0.493
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:area"] = (
            k_factor_vt
            * (
                0.19
                * 1.014
                * 1.089
                * (
                    (1 + has_t_tail) ** 0.5
                    * (sizing_factor_ultimate * mtow) ** 0.363
                    * mach_h**0.601
                    * lp_vt**-0.726
                    * (1 + rudder_chord_ratio) ** 0.217
                    * ar_vt**0.337
                    * (1 + taper_vt) ** 0.363
                    * np.cos(sweep_25_vt) ** -0.484
                )
                ** 1.014
                * area_vt**0.104246
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
        ] = k_factor_vt * (
            -0.19
            * 1.014
            * 0.726
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
            * lp_vt ** (-1.736164)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:geometry:vertical_tail:rudder:chord_ratio",
        ] = k_factor_vt * (
            0.19
            * 1.014
            * 0.217
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
            * (1 + rudder_chord_ratio) ** (-0.779962)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:sweep_25"
        ] = k_factor_vt * (
            0.19
            * 1.014
            * 0.484
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
            )
            ** 1.014
            * np.cos(sweep_25_vt) ** (-1.490776)
            * np.sin(sweep_25_vt)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:aspect_ratio"
        ] = k_factor_vt * (
            0.19
            * 1.014
            * 0.337
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
            * ar_vt ** (-0.658282)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:taper_ratio"
        ] = k_factor_vt * (
            0.19
            * 1.014
            * 0.363
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
            * (1 + taper_vt) ** (-0.631918)
        )

        partials["data:weight:airframe:vertical_tail:mass", "data:TLAR:v_max_sl"] = k_factor_vt * (
            0.19
            * 1.014
            * 0.601
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
            * mach_h ** (-0.390586)
            / atm0.speed_of_sound
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:weight:airframe:vertical_tail:k_factor"
        ] = (
            0.19
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
        )

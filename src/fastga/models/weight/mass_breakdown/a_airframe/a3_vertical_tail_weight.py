"""
Estimation of tail weight.
"""
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

from stdatm import Atmosphere
import fastoad.api as oad

from .constants import SUBMODEL_VERTICAL_TAIL_MASS

oad.RegisterSubmodel.active_models[
    SUBMODEL_VERTICAL_TAIL_MASS
] = "fastga.submodel.weight.mass.airframe.vertical_tail.legacy"


@oad.RegisterSubmodel(
    SUBMODEL_VERTICAL_TAIL_MASS, "fastga.submodel.weight.mass.airframe.vertical_tail.legacy"
)
class ComputeVerticalTailWeight(om.ExplicitComponent):
    """
    Weight estimation for tail weight

    Based on a statistical analysis. See :cite:`raymer:2012` but can also be found in
    :cite:`gudmundsson:2013`
    """

    def setup(self):

        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:vertical_tail:k_factor", val=1.0)
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")

        self.add_input("data:geometry:has_T_tail", val=np.nan)
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:vertical_tail:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)

        self.add_output("data:weight:airframe:vertical_tail:mass", units="lb")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        v_cruise_ktas = inputs["data:TLAR:v_cruise"]
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]

        rho_cruise = Atmosphere(cruise_alt, altitude_in_feet=True).density
        dynamic_pressure = 1.0 / 2.0 * rho_cruise * v_cruise_ktas ** 2.0 * 0.0208854
        # In lb/ft2

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
                * dynamic_pressure ** 0.122
                * area_vt ** 0.873
                * (100.0 * t_c_vt / np.cos(sweep_25_vt * np.pi / 180.0)) ** -0.49
                * (ar_vt / (np.cos(sweep_25_vt * np.pi / 180.0)) ** 2.0) ** 0.357
                * taper_vt ** 0.039
            )
        )
        # Mass formula in lb

        outputs["data:weight:airframe:vertical_tail:mass"] = (
            a32 * inputs["data:weight:airframe:vertical_tail:k_factor"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        v_cruise_ktas = inputs["data:TLAR:v_cruise"]
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]

        rho_cruise = Atmosphere(cruise_alt, altitude_in_feet=True).density
        dynamic_pressure = 1.0 / 2.0 * rho_cruise * v_cruise_ktas ** 2.0 * 0.0208854
        # In lb/ft2

        has_t_tail = inputs["data:geometry:has_T_tail"]
        area_vt = inputs["data:geometry:vertical_tail:area"]
        t_c_vt = inputs["data:geometry:vertical_tail:thickness_ratio"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]

        k_factor = inputs["data:weight:airframe:vertical_tail:k_factor"]

        a32 = (
            0.073
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** 0.376
                * dynamic_pressure ** 0.122
                * area_vt ** 0.873
                * (100.0 * t_c_vt / np.cos(sweep_25_vt * np.pi / 180.0)) ** -0.49
                * (ar_vt / (np.cos(sweep_25_vt * np.pi / 180.0)) ** 2.0) ** 0.357
                * taper_vt ** 0.039
            )
        )
        # Mass formula in lb

        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor * (
            (
                0.376
                * area_vt ** 0.8730
                * mtow
                * taper_vt ** 0.0390
                * (ar_vt / np.cos(0.017453 * sweep_25_vt) ** 2) ** 0.3570
                * (0.0146 * has_t_tail + 0.073)
                * (0.010443 * rho_cruise * v_cruise_ktas ** 2) ** 0.1220
            )
            / (
                (mtow * sizing_factor_ultimate) ** 0.6240
                * ((100.0 * t_c_vt) / np.cos(0.017453 * sweep_25_vt)) ** 0.4900
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:weight:aircraft:MTOW"
        ] = k_factor * (
            (
                0.376
                * area_vt ** 0.8730
                * sizing_factor_ultimate
                * taper_vt ** 0.0390
                * (ar_vt / np.cos(0.017453 * sweep_25_vt) ** 2) ** 0.3570
                * (0.0146 * has_t_tail + 0.073)
                * (0.010443 * rho_cruise * v_cruise_ktas ** 2) ** 0.1220
            )
            / (
                (mtow * sizing_factor_ultimate) ** 0.6240
                * ((100.0 * t_c_vt) / np.cos(0.017453 * sweep_25_vt)) ** 0.4900
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:TLAR:v_cruise"] = k_factor * (
            (
                0.002548
                * area_vt ** 0.8730
                * rho_cruise
                * taper_vt ** 0.0390
                * v_cruise_ktas
                * (mtow * sizing_factor_ultimate) ** 0.3760
                * (ar_vt / np.cos(0.017453 * sweep_25_vt) ** 2) ** 0.3570
                * (0.0146 * has_t_tail + 0.073)
            )
            / (
                ((100.0 * t_c_vt) / np.cos(0.017453 * sweep_25_vt)) ** 0.4900
                * (0.010443 * rho_cruise * v_cruise_ktas ** 2) ** 0.8780
            )
        )
        d_a32_d_rho_cruise = (
            0.001274
            * area_vt ** 0.8730
            * taper_vt ** 0.0390
            * v_cruise_ktas ** 2
            * (mtow * sizing_factor_ultimate) ** 0.3760
            * (ar_vt / np.cos(0.017453 * sweep_25_vt) ** 2) ** 0.3570
            * (0.0146 * has_t_tail + 0.073)
        ) / (
            ((100.0 * t_c_vt) / np.cos(0.017453 * sweep_25_vt)) ** 0.4900
            * (0.010443 * rho_cruise * v_cruise_ktas ** 2) ** 0.8780
        )
        d_rho_cruise_d_cruise_alt = 2.3e-6  # lb/ft**4
        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:mission:sizing:main_route:cruise:altitude",
        ] = k_factor * (d_a32_d_rho_cruise * d_rho_cruise_d_cruise_alt)
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:has_T_tail"
        ] = k_factor * (
            (
                0.0146
                * area_vt ** 0.8730
                * taper_vt ** 0.0390
                * (mtow * sizing_factor_ultimate) ** 0.3760
                * (ar_vt / np.cos(0.017453 * sweep_25_vt) ** 2) ** 0.3570
                * (0.010443 * rho_cruise * v_cruise_ktas ** 2) ** 0.1220
            )
            / ((100.0 * t_c_vt) / np.cos(0.017453 * sweep_25_vt)) ** 0.4900
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:area"
        ] = k_factor * (
            (
                0.873
                * taper_vt ** 0.0390
                * (mtow * sizing_factor_ultimate) ** 0.3760
                * (ar_vt / np.cos(0.017453 * sweep_25_vt) ** 2) ** 0.3570
                * (0.0146 * has_t_tail + 0.073)
                * (0.010443 * rho_cruise * v_cruise_ktas ** 2) ** 0.1220
            )
            / (
                area_vt ** (127 / 1000)
                * ((100.0 * t_c_vt) / np.cos(0.017453 * sweep_25_vt)) ** 0.4900
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:thickness_ratio"
        ] = k_factor * (
            -(
                49.0
                * area_vt ** 0.8730
                * taper_vt ** 0.0390
                * (mtow * sizing_factor_ultimate) ** 0.3760
                * (ar_vt / np.cos(0.01745329252 * sweep_25_vt) ** 2) ** 0.3570
                * (0.0146 * has_t_tail + 0.073)
                * (0.0104427 * rho_cruise * v_cruise_ktas ** 2) ** 0.1220
            )
            / (
                np.cos(0.01745329252 * sweep_25_vt)
                * ((100.0 * t_c_vt) / np.cos(0.01745329252 * sweep_25_vt)) ** 1.4900
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:sweep_25"
        ] = k_factor * (
            (
                0.012462
                * ar_vt
                * area_vt ** 0.8730
                * taper_vt ** 0.0390
                * np.sin(0.017453 * sweep_25_vt)
                * (mtow * sizing_factor_ultimate) ** 0.3760
                * (0.0146 * has_t_tail + 0.073)
                * (0.010443 * rho_cruise * v_cruise_ktas ** 2) ** 0.1220
            )
            / (
                np.cos(0.017453 * sweep_25_vt) ** 3
                * (ar_vt / np.cos(0.017453 * sweep_25_vt) ** 2) ** 0.6430
                * ((100.0 * t_c_vt) / np.cos(0.017453 * sweep_25_vt)) ** 0.4900
            )
            - (
                0.85521
                * area_vt ** 0.8730
                * t_c_vt
                * taper_vt ** 0.0390
                * np.sin(0.017453 * sweep_25_vt)
                * (mtow * sizing_factor_ultimate) ** 0.3760
                * (ar_vt / np.cos(0.017453 * sweep_25_vt) ** 2) ** 0.3570
                * (0.0146 * has_t_tail + 0.073)
                * (0.010443 * rho_cruise * v_cruise_ktas ** 2) ** 0.1220
            )
            / (
                np.cos(0.017453 * sweep_25_vt) ** 2
                * ((100.0 * t_c_vt) / np.cos(0.017453 * sweep_25_vt)) ** 1.4900
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:aspect_ratio"
        ] = k_factor * (
            (
                0.357
                * area_vt ** 0.8730
                * taper_vt ** 0.0390
                * (mtow * sizing_factor_ultimate) ** 0.3760
                * (0.0146 * has_t_tail + 0.073)
                * (0.010443 * rho_cruise * v_cruise_ktas ** 2) ** 0.1220
            )
            / (
                np.cos(0.017453 * sweep_25_vt) ** 2
                * (ar_vt / np.cos(0.017453 * sweep_25_vt) ** 2) ** 0.6430
                * ((100.0 * t_c_vt) / np.cos(0.017453 * sweep_25_vt)) ** 0.4900
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:taper_ratio"
        ] = k_factor * (
            (
                0.039
                * area_vt ** 0.8730
                * (mtow * sizing_factor_ultimate) ** 0.3760
                * (ar_vt / np.cos(0.017453 * sweep_25_vt) ** 2) ** 0.3570
                * (0.0146 * has_t_tail + 0.073)
                * (0.010443 * rho_cruise * v_cruise_ktas ** 2) ** 0.1220
            )
            / (taper_vt ** 0.9610 * ((100.0 * t_c_vt) / np.cos(0.017453 * sweep_25_vt)) ** 0.4900)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:weight:airframe:vertical_tail:k_factor"
        ] = a32

"""
Python module for htp weight calculation, part of the tail mass computation.
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

from .constants import (
    SERVICE_HTP_MASS,
    SUBMODEL_HTP_MASS_LEGACY,
    SUBMODEL_HTP_MASS_GD,
    SUBMODEL_HTP_MASS_TORENBEEK,
)


@oad.RegisterSubmodel(SERVICE_HTP_MASS, SUBMODEL_HTP_MASS_LEGACY)
class ComputeHTPWeight(om.ExplicitComponent):
    """
    Weight estimation for htp weight

    Based on a statistical analysis. See :cite:`raymer:2012` but can also be found in
    :cite:`gudmundsson:2013`
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:horizontal_tail:k_factor", val=1.0)
        self.add_input("data:weight:airframe:vertical_tail:k_factor", val=1.0)
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")

        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:horizontal_tail:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)

        self.add_output("data:weight:airframe:horizontal_tail:mass", units="lb")

        self.declare_partials(
            of="data:weight:airframe:horizontal_tail:mass",
            wrt=[
                "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
                "data:weight:aircraft:MTOW",
                "data:weight:airframe:horizontal_tail:k_factor",
                "data:TLAR:v_cruise",
                "data:geometry:horizontal_tail:area",
                "data:geometry:horizontal_tail:thickness_ratio",
                "data:geometry:horizontal_tail:sweep_25",
                "data:geometry:horizontal_tail:aspect_ratio",
                "data:geometry:horizontal_tail:taper_ratio",
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

        area_ht = inputs["data:geometry:horizontal_tail:area"]
        t_c_ht = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        ar_ht = inputs["data:geometry:horizontal_tail:aspect_ratio"]
        taper_ht = inputs["data:geometry:horizontal_tail:taper_ratio"]

        rho_cruise = AtmosphereWithPartials(cruise_alt, altitude_in_feet=True).density
        dynamic_pressure = 1.0 / 2.0 * rho_cruise * v_cruise_ktas**2.0 * 0.0208854
        # In lb/ft2

        a31 = 0.016 * (
            (sizing_factor_ultimate * mtow) ** 0.414
            * dynamic_pressure**0.168
            * area_ht**0.896
            * (100.0 * t_c_ht / np.cos(sweep_25_ht)) ** -0.12
            * (ar_ht / np.cos(sweep_25_ht) ** 2.0) ** 0.043
            * taper_ht**-0.02
        )
        # Mass formula in lb

        outputs["data:weight:airframe:horizontal_tail:mass"] = (
            a31 * inputs["data:weight:airframe:horizontal_tail:k_factor"]
        )

    # pylint: disable=missing-function-docstring, unused-argument, too-many-local-variables
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        v_cruise_ktas = inputs["data:TLAR:v_cruise"]
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]
        atm_cruise = AtmosphereWithPartials(cruise_alt)

        area_ht = inputs["data:geometry:horizontal_tail:area"]
        t_c_ht = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        ar_ht = inputs["data:geometry:horizontal_tail:aspect_ratio"]
        taper_ht = inputs["data:geometry:horizontal_tail:taper_ratio"]

        k_factor_ht = inputs["data:weight:airframe:horizontal_tail:k_factor"]

        rho_cruise = atm_cruise.density
        dynamic_pressure = 1.0 / 2.0 * rho_cruise * v_cruise_ktas**2.0 * 0.0208854

        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor_ht * (
            0.016
            * (
                0.414
                * (sizing_factor_ultimate * mtow) ** -0.586
                * mtow
                * dynamic_pressure**0.168
                * area_ht**0.896
                * (100.0 * t_c_ht / np.cos(sweep_25_ht)) ** -0.12
                * (ar_ht / np.cos(sweep_25_ht) ** 2.0) ** 0.043
                * taper_ht**-0.02
            )
        )
        partials["data:weight:airframe:horizontal_tail:mass", "data:weight:aircraft:MTOW"] = (
            k_factor_ht
            * (
                0.016
                * (
                    0.414
                    * (sizing_factor_ultimate * mtow) ** -0.586
                    * sizing_factor_ultimate
                    * dynamic_pressure**0.168
                    * area_ht**0.896
                    * (100.0 * t_c_ht / np.cos(sweep_25_ht)) ** -0.12
                    * (ar_ht / np.cos(sweep_25_ht) ** 2.0) ** 0.043
                    * taper_ht**-0.02
                )
            )
        )
        partials["data:weight:airframe:horizontal_tail:mass", "data:TLAR:v_cruise"] = (
            k_factor_ht
            * 0.016
            * 0.336
            * (
                (sizing_factor_ultimate * mtow) ** 0.414
                * (0.5 * rho_cruise * 0.0208854) ** 0.168
                * v_cruise_ktas ** (-0.664)
                * area_ht**0.896
                * (100.0 * t_c_ht / np.cos(sweep_25_ht)) ** -0.12
                * (ar_ht / np.cos(sweep_25_ht) ** 2.0) ** 0.043
                * taper_ht**-0.02
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:area"
        ] = k_factor_ht * (
            0.016
            * 0.896
            * (
                (sizing_factor_ultimate * mtow) ** 0.414
                * dynamic_pressure**0.168
                * area_ht ** (-0.104)
                * (100.0 * t_c_ht / np.cos(sweep_25_ht)) ** -0.12
                * (ar_ht / np.cos(sweep_25_ht) ** 2.0) ** 0.043
                * taper_ht**-0.02
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:geometry:horizontal_tail:thickness_ratio",
        ] = k_factor_ht * (
            0.016
            * (
                (sizing_factor_ultimate * mtow) ** 0.414
                * dynamic_pressure**0.168
                * area_ht**0.896
                * (100.0 / np.cos(sweep_25_ht)) ** -0.12
                * (-0.12 * t_c_ht**-1.12)
                * (ar_ht / np.cos(sweep_25_ht) ** 2.0) ** 0.043
                * taper_ht**-0.02
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:sweep_25"
        ] = k_factor_ht * (
            -0.016
            * 0.034
            * (
                (sizing_factor_ultimate * mtow) ** 0.414
                * dynamic_pressure**0.168
                * area_ht**0.896
                * (100.0 * t_c_ht) ** -0.12
                * ar_ht**0.043
                * taper_ht**-0.02
                * np.cos(sweep_25_ht) ** (-0.966)
                * np.sin(sweep_25_ht)
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:geometry:horizontal_tail:aspect_ratio",
        ] = k_factor_ht * (
            0.016
            * 0.043
            * (
                (sizing_factor_ultimate * mtow) ** 0.414
                * dynamic_pressure**0.168
                * area_ht**0.896
                * (100.0 * t_c_ht / np.cos(sweep_25_ht)) ** -0.12
                * (1 / np.cos(sweep_25_ht) ** 2.0) ** 0.043
                * ar_ht ** (-0.957)
                * taper_ht**-0.02
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:taper_ratio"
        ] = k_factor_ht * (
            -0.016
            * 0.02
            * (
                (sizing_factor_ultimate * mtow) ** 0.414
                * dynamic_pressure**0.168
                * area_ht**0.896
                * (100.0 * t_c_ht / np.cos(sweep_25_ht)) ** -0.12
                * (ar_ht / np.cos(sweep_25_ht) ** 2.0) ** 0.043
                * taper_ht**-1.02
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:weight:airframe:horizontal_tail:k_factor",
        ] = 0.016 * (
            (sizing_factor_ultimate * mtow) ** 0.414
            * dynamic_pressure**0.168
            * area_ht**0.896
            * (100.0 * t_c_ht / np.cos(sweep_25_ht)) ** -0.12
            * (ar_ht / np.cos(sweep_25_ht) ** 2.0) ** 0.043
            * taper_ht**-0.02
        )

        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:mission:sizing:main_route:cruise:altitude",
        ] = k_factor_ht * (
            0.016
            * (
                (sizing_factor_ultimate * mtow) ** 0.414
                * (0.5 * v_cruise_ktas**2.0 * 0.0208854) ** 0.168
                * 0.168
                * rho_cruise ** (-0.832)
                * atm_cruise.partial_density_altitude
                * area_ht**0.896
                * (100.0 * t_c_ht / np.cos(sweep_25_ht)) ** -0.12
                * (ar_ht / np.cos(sweep_25_ht) ** 2.0) ** 0.043
                * taper_ht**-0.02
            )
        )


@oad.RegisterSubmodel(SERVICE_HTP_MASS, SUBMODEL_HTP_MASS_GD)
class ComputeHTPWeightGD(om.ExplicitComponent):
    """
    Weight estimation for htp weight

    Based on a statistical analysis. See :cite:`roskampart5:1985` traditionally used on
    commercial aircraft but found to work fairly well on high performance GA aircraft.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:horizontal_tail:k_factor", val=1.0)

        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:horizontal_tail:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="ft")
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="ft")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="ft")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="ft"
        )

        self.add_output("data:weight:airframe:horizontal_tail:mass", units="lb")

        self.declare_partials(
            of="data:weight:airframe:horizontal_tail:mass",
            wrt=[
                "data:weight:aircraft:MTOW",
                "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
                "data:geometry:horizontal_tail:area",
                "data:geometry:horizontal_tail:thickness_ratio",
                "data:geometry:horizontal_tail:root:chord",
                "data:geometry:horizontal_tail:span",
                "data:geometry:horizontal_tail:MAC:length",
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:weight:airframe:horizontal_tail:k_factor",
            ],
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument, too-many-local-variables
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]

        area_ht = inputs["data:geometry:horizontal_tail:area"]
        span_ht = inputs["data:geometry:horizontal_tail:span"]
        t_c_ht = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        root_chord_ht = inputs["data:geometry:horizontal_tail:root:chord"]
        root_thickness = t_c_ht * root_chord_ht
        mac_ht = inputs["data:geometry:horizontal_tail:MAC:length"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]

        a31 = (
            0.0034
            * (
                (sizing_factor_ultimate * mtow) ** 0.813
                * area_ht**0.584
                * (span_ht / root_thickness) ** 0.033
                * (mac_ht / lp_ht) ** 0.28
            )
            ** 0.915
        )
        # Mass formula in lb

        outputs["data:weight:airframe:horizontal_tail:mass"] = (
            a31 * inputs["data:weight:airframe:horizontal_tail:k_factor"]
        )

    # pylint: disable=missing-function-docstring, unused-argument, too-many-local-variables
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]

        area_ht = inputs["data:geometry:horizontal_tail:area"]
        span_ht = inputs["data:geometry:horizontal_tail:span"]
        t_c_ht = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        root_chord_ht = inputs["data:geometry:horizontal_tail:root:chord"]
        mac_ht = inputs["data:geometry:horizontal_tail:MAC:length"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        k_factor_ht = inputs["data:weight:airframe:horizontal_tail:k_factor"]

        root_thickness = t_c_ht * root_chord_ht

        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor_ht * (
            0.0034
            * 0.915
            * 0.813
            * (area_ht**0.584 * (span_ht / root_thickness) ** 0.033 * (mac_ht / lp_ht) ** 0.28)
            ** 0.915
            * (sizing_factor_ultimate * mtow) ** (-0.256105)
            * mtow
        )
        partials["data:weight:airframe:horizontal_tail:mass", "data:weight:aircraft:MTOW"] = (
            k_factor_ht
            * (
                0.0034
                * 0.915
                * 0.813
                * (area_ht**0.584 * (span_ht / root_thickness) ** 0.033 * (mac_ht / lp_ht) ** 0.28)
                ** 0.915
                * (sizing_factor_ultimate * mtow) ** (-0.256105)
                * sizing_factor_ultimate
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:area"
        ] = k_factor_ht * (
            0.0034
            * 0.915
            * 0.584
            * (
                (sizing_factor_ultimate * mtow) ** 0.813
                * (span_ht / root_thickness) ** 0.033
                * (mac_ht / lp_ht) ** 0.28
            )
            ** 0.915
            * area_ht ** (-0.46564)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:span"
        ] = k_factor_ht * (
            0.0034
            * 0.915
            * 0.033
            * (
                (sizing_factor_ultimate * mtow) ** 0.813
                * area_ht**0.584
                * root_thickness ** (-0.033)
                * (mac_ht / lp_ht) ** 0.28
            )
            ** 0.915
            * span_ht ** (-0.969805)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:geometry:horizontal_tail:thickness_ratio",
        ] = k_factor_ht * (
            -0.0034
            * 0.915
            * 0.033
            * (
                (sizing_factor_ultimate * mtow) ** 0.813
                * area_ht**0.584
                * (span_ht / root_chord_ht) ** 0.033
                * (mac_ht / lp_ht) ** 0.28
            )
            ** 0.915
            * t_c_ht ** (-1.030195)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:root:chord"
        ] = k_factor_ht * (
            -0.0034
            * 0.915
            * 0.033
            * (
                (sizing_factor_ultimate * mtow) ** 0.813
                * area_ht**0.584
                * (span_ht / t_c_ht) ** 0.033
                * (mac_ht / lp_ht) ** 0.28
            )
            ** 0.915
            * root_chord_ht ** (-1.030195)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:MAC:length"
        ] = k_factor_ht * (
            0.0034
            * 0.915
            * 0.28
            * (
                (sizing_factor_ultimate * mtow) ** 0.813
                * area_ht**0.584
                * (span_ht / root_thickness) ** 0.033
                * lp_ht**-0.28
            )
            ** 0.915
            * mac_ht ** (-0.7438)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
        ] = k_factor_ht * (
            -0.0034
            * 0.915
            * 0.28
            * (
                (sizing_factor_ultimate * mtow) ** 0.813
                * area_ht**0.584
                * (span_ht / root_thickness) ** 0.033
                * mac_ht**0.28
            )
            ** 0.915
            * lp_ht ** (-1.2562)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:weight:airframe:horizontal_tail:k_factor",
        ] = (
            0.0034
            * (
                (sizing_factor_ultimate * mtow) ** 0.813
                * area_ht**0.584
                * (span_ht / root_thickness) ** 0.033
                * (mac_ht / lp_ht) ** 0.28
            )
            ** 0.915
        )


@oad.RegisterSubmodel(SERVICE_HTP_MASS, SUBMODEL_HTP_MASS_TORENBEEK)
class ComputeHTPWeightTorenbeek(om.ExplicitComponent):
    """
    Weight estimation for htp weight

    Based on a statistical analysis. See :cite:`roskampart5:1985` traditionally used on
    commercial aircraft but found to work fairly well on high performance GA aircraft. Should
    only be used with aircraft having a diving speed above 250 KEAS.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:weight:airframe:horizontal_tail:k_factor", val=1.0)
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:mission:sizing:cs23:characteristic_speed:vd", val=np.nan, units="kn")

        self.add_output("data:weight:airframe:horizontal_tail:mass", units="lb")

        self.declare_partials(
            of="data:weight:airframe:horizontal_tail:mass",
            wrt=[
                "data:geometry:horizontal_tail:area",
                "data:weight:airframe:horizontal_tail:k_factor",
                "data:mission:sizing:cs23:characteristic_speed:vd",
                "data:geometry:horizontal_tail:sweep_25",
            ],
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument, too-many-local-variables
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        area_ht = inputs["data:geometry:horizontal_tail:area"]
        vd = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]
        sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]

        a31 = area_ht * (3.81 * area_ht**0.2 * vd / (1000.0 * np.cos(sweep_25)) - 0.287)
        # Mass formula in lb

        outputs["data:weight:airframe:horizontal_tail:mass"] = (
            a31 * inputs["data:weight:airframe:horizontal_tail:k_factor"]
        )

    # pylint: disable=missing-function-docstring, unused-argument, too-many-local-variables
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        area_ht = inputs["data:geometry:horizontal_tail:area"]
        vd = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]
        sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]
        k_factor_ht = inputs["data:weight:airframe:horizontal_tail:k_factor"]

        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:area"
        ] = k_factor_ht * ((0.004572 * area_ht**0.2 * vd) / np.cos(sweep_25) - 0.287)
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:mission:sizing:cs23:characteristic_speed:vd",
        ] = k_factor_ht * ((0.00381 * area_ht**1.2) / np.cos(sweep_25))
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:sweep_25"
        ] = k_factor_ht * (
            (0.00381 * area_ht**1.2 * vd * np.sin(sweep_25)) / np.cos(sweep_25) ** 2.0
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:weight:airframe:horizontal_tail:k_factor",
        ] = area_ht * (3.81 * area_ht**0.2 * vd / (1000.0 * np.cos(sweep_25)) - 0.287)

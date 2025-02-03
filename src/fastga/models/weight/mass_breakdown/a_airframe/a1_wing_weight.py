"""
Python module for wing weight calculation, part of the airframe mass computation.
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

from .constants import SERVICE_WING_MASS, SUBMODEL_WING_MASS_LEGACY

oad.RegisterSubmodel.active_models[SERVICE_WING_MASS] = SUBMODEL_WING_MASS_LEGACY


@oad.RegisterSubmodel(SERVICE_WING_MASS, SUBMODEL_WING_MASS_LEGACY)
class ComputeWingWeight(om.ExplicitComponent):
    """
    Wing weight estimation

    Based on a statistical analysis. See :cite:`nicolai:2010` but can also be found in
    :cite:`gudmundsson:2013`.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:wing:k_factor", val=1.0)
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:TLAR:v_max_sl", val=np.nan, units="kn")

        self.add_output("data:weight:airframe:wing:mass", units="lb")

        self.declare_partials("*", "*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        wing_area = inputs["data:geometry:wing:area"]
        taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        thickness_ratio = inputs["data:geometry:wing:thickness_ratio"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        aspect_ratio = inputs["data:geometry:wing:aspect_ratio"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        v_max_sl = inputs["data:TLAR:v_max_sl"]

        a1 = (
            96.948
            * (
                (mtow * sizing_factor_ultimate / 10.0**5.0) ** 0.65
                * (aspect_ratio / (np.cos(sweep_25) ** 2.0)) ** 0.57
                * (wing_area / 100.0) ** 0.61
                * ((1.0 + taper_ratio) / (2.0 * thickness_ratio)) ** 0.36
                * (1.0 + v_max_sl / 500.0) ** 0.5
            )
            ** 0.993
        )  # mass formula in lb

        outputs["data:weight:airframe:wing:mass"] = (
            a1 * inputs["data:weight:airframe:wing:k_factor"]
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        wing_area = inputs["data:geometry:wing:area"]
        taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        thickness_ratio = inputs["data:geometry:wing:thickness_ratio"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        aspect_ratio = inputs["data:geometry:wing:aspect_ratio"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        v_max_sl = inputs["data:TLAR:v_max_sl"]
        k_factor = inputs["data:weight:airframe:wing:k_factor"]

        partials[
            "data:weight:airframe:wing:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = (
            96.948
            * 0.64545
            * (
                (mtow / 10.0**5.0) ** 0.65
                * (aspect_ratio / (np.cos(sweep_25) ** 2.0)) ** 0.57
                * (wing_area / 100.0) ** 0.61
                * ((1.0 + taper_ratio) / (2.0 * thickness_ratio)) ** 0.36
                * (1.0 + v_max_sl / 500.0) ** 0.5
            )
            ** 0.993
            * sizing_factor_ultimate**-0.35455
        ) * k_factor
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:area"] = (
            96.948
            * 0.60573
            * (
                (mtow * sizing_factor_ultimate / 10.0**5.0) ** 0.65
                * (aspect_ratio / (np.cos(sweep_25) ** 2.0)) ** 0.57
                * 0.01**0.61
                * ((1.0 + taper_ratio) / (2.0 * thickness_ratio)) ** 0.36
                * (1.0 + v_max_sl / 500.0) ** 0.5
            )
            ** 0.993
            * wing_area**-0.39427
        ) * k_factor
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:taper_ratio"] = (
            96.948
            * 0.35748
            * (
                (mtow * sizing_factor_ultimate / 10.0**5.0) ** 0.65
                * (aspect_ratio / (np.cos(sweep_25) ** 2.0)) ** 0.57
                * (wing_area / 100.0) ** 0.61
                * (1.0 + v_max_sl / 500.0) ** 0.5
                * 1
                / (2.0 * thickness_ratio) ** 0.36
            )
            ** 0.993
            * (1.0 + taper_ratio) ** -0.64252
        ) * k_factor
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:thickness_ratio"] = (
            -96.948
            * 0.35748
            * (
                (mtow * sizing_factor_ultimate / 10.0**5.0) ** 0.65
                * (aspect_ratio / (np.cos(sweep_25) ** 2.0)) ** 0.57
                * (wing_area / 100.0) ** 0.61
                * ((1.0 + taper_ratio) / 2.0) ** 0.36
                * (1.0 + v_max_sl / 500.0) ** 0.5
            )
            ** 0.993
            * thickness_ratio**-1.35748
        ) * k_factor
        partials["data:weight:airframe:wing:mass", "data:weight:aircraft:MTOW"] = (
            96.948
            * 0.64545
            * (
                (sizing_factor_ultimate / 10.0**5.0) ** 0.65
                * (aspect_ratio / (np.cos(sweep_25) ** 2.0)) ** 0.57
                * (wing_area / 100.0) ** 0.61
                * ((1.0 + taper_ratio) / (2.0 * thickness_ratio)) ** 0.36
                * (1.0 + v_max_sl / 500.0) ** 0.5
            )
            ** 0.993
            * mtow**-0.35455
        ) * k_factor
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:aspect_ratio"] = (
            96.948
            * 0.56601
            * (
                (mtow * sizing_factor_ultimate / 10.0**5.0) ** 0.65
                * (np.cos(sweep_25) ** 2.0) ** 0.57
                * (wing_area / 100.0) ** 0.61
                * ((1.0 + taper_ratio) / (2.0 * thickness_ratio)) ** 0.36
                * (1.0 + v_max_sl / 500.0) ** 0.5
            )
            ** 0.993
            * aspect_ratio**-0.43399
        ) * k_factor
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:sweep_25"] = (
            96.948
            * 1.13202
            * (
                (mtow * sizing_factor_ultimate / 10.0**5.0) ** 0.65
                * aspect_ratio**0.57
                * (wing_area / 100.0) ** 0.61
                * ((1.0 + taper_ratio) / (2.0 * thickness_ratio)) ** 0.36
                * (1.0 + v_max_sl / 500.0) ** 0.5
            )
            ** 0.993
            * np.cos(sweep_25) ** -2.13202
            * np.sin(sweep_25)
        ) * k_factor
        partials["data:weight:airframe:wing:mass", "data:TLAR:v_max_sl"] = (
            96.948
            * 0.4965
            / 500.0
            * (
                (mtow * sizing_factor_ultimate / 10.0**5.0) ** 0.65
                * (aspect_ratio / (np.cos(sweep_25) ** 2.0)) ** 0.57
                * (wing_area / 100.0) ** 0.61
                * ((1.0 + taper_ratio) / (2.0 * thickness_ratio)) ** 0.36
            )
            ** 0.993
            * (1.0 + v_max_sl / 500.0) ** -0.5035
        ) * k_factor
        partials["data:weight:airframe:wing:mass", "data:weight:airframe:wing:k_factor"] = (
            96.948
            * (
                (mtow * sizing_factor_ultimate / 10.0**5.0) ** 0.65
                * (aspect_ratio / (np.cos(sweep_25) ** 2.0)) ** 0.57
                * (wing_area / 100.0) ** 0.61
                * ((1.0 + taper_ratio) / (2.0 * thickness_ratio)) ** 0.36
                * (1.0 + v_max_sl / 500.0) ** 0.5
            )
            ** 0.993
        )

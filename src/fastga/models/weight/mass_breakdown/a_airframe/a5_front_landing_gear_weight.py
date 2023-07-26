"""
Estimation of landing gear weight.
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

import fastoad.api as oad

from .constants import SUBMODEL_FRONT_LANDING_GEAR_MASS

oad.RegisterSubmodel.active_models[
    SUBMODEL_FRONT_LANDING_GEAR_MASS
] = "fastga.submodel.weight.mass.airframe.front_landing_gear.legacy"


@oad.RegisterSubmodel(
    SUBMODEL_FRONT_LANDING_GEAR_MASS,
    "fastga.submodel.weight.mass.airframe.front_landing_gear.legacy",
)
class ComputeFrontLandingGearWeight(om.ExplicitComponent):
    """
    Weight estimation for landing gears

    Based on a statistical analysis. See :cite:`wells:2017` for the formula and
    :cite:`raymer:2012` for the weight reduction factor for non retractable landing gears
    """

    def setup(self):

        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="lb")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:geometry:landing_gear:height", val=np.nan, units="inch")
        self.add_input("data:geometry:landing_gear:type", val=np.nan)
        self.add_input("data:geometry:wing_configuration", val=np.nan)

        self.add_output("data:weight:airframe:landing_gear:front:mass", units="lb")

        self.declare_partials(
            of="*",
            wrt=[
                "data:weight:aircraft:MLW",
                "data:weight:aircraft:MTOW",
                "data:geometry:landing_gear:height",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mlw = inputs["data:weight:aircraft:MLW"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        lg_height = inputs["data:geometry:landing_gear:height"]
        is_retractable = inputs["data:geometry:landing_gear:type"]
        wing_config = inputs["data:geometry:wing_configuration"]

        carrier_based = 0.0
        aircraft_type = 0.0  # One for fighter/attack aircraft

        # To prevent using obstruse data we put this failsafe here
        # TODO : Find a better way to do this
        if mlw < mtow / 2.0:
            mlw = mtow

        mlg_weight = (0.0117 - aircraft_type * 0.0012) * mlw ** 0.95 * lg_height ** 0.43
        nlg_weight = (
            (0.048 - aircraft_type * 0.008)
            * mlw ** 0.67
            * lg_height ** 0.43
            * (1.0 + 0.8 * carrier_based)
        )

        if not is_retractable:
            weight_reduction = 1.4 * mtow / 100.0
            weight_reduction_factor = (mlg_weight + nlg_weight - weight_reduction) / (
                mlg_weight + nlg_weight
            )
        else:
            weight_reduction_factor = 1.0

        if wing_config == 3.0:
            nlg_weight *= 1.08

        outputs["data:weight:airframe:landing_gear:front:mass"] = (
            nlg_weight * weight_reduction_factor
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        mlw = inputs["data:weight:aircraft:MLW"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        lg_height = inputs["data:geometry:landing_gear:height"]
        is_retractable = inputs["data:geometry:landing_gear:type"]
        wing_config = inputs["data:geometry:wing_configuration"]

        carrier_based = 0.0
        aircraft_type = 0.0  # One for fighter/attack aircraft

        if mlw < mtow / 2.0:
            mlw = mtow

            mlg_weight = (0.0117 - aircraft_type * 0.0012) * mlw ** 0.95 * lg_height ** 0.43
            nlg_weight = (
                (0.048 - aircraft_type * 0.008)
                * mlw ** 0.67
                * lg_height ** 0.43
                * (1.0 + 0.8 * carrier_based)
            )
            lg_weight = mlg_weight + nlg_weight

            d_nlg_weight_d_mtow = (
                -(
                    0.67
                    * lg_height ** 0.43
                    * (0.008 * aircraft_type - 0.048)
                    * (0.8 * carrier_based + 1.0)
                )
                / mtow ** 0.33
            )
            d_mlg_weight_d_mtow = (
                -(0.95 * lg_height ** 0.43 * (0.0012 * aircraft_type - 0.0117)) / mtow ** 0.05
            )
            d_lg_weight_d_mtow = d_nlg_weight_d_mtow + d_mlg_weight_d_mtow

            d_nlg_height_d_lg_height = (
                -(
                    0.43
                    * mtow ** 0.67
                    * (0.008 * aircraft_type - 0.048)
                    * (0.8 * carrier_based + 1.0)
                )
                / lg_height ** 0.57
            )
            d_mlg_height_d_lg_height = (
                -(0.43 * mtow ** 0.95 * (0.0012 * aircraft_type - 0.0117)) / lg_height ** 0.57
            )

            if not is_retractable:
                weight_reduction = 1.4 * mtow / 100.0
                weight_reduction_factor = (mlg_weight + nlg_weight - weight_reduction) / (
                    mlg_weight + nlg_weight
                )

                d_weight_reduction_d_mtow = 1.4 / 100.0
                d_weight_reduction_factor_d_mtow = (
                    -(d_weight_reduction_d_mtow * lg_weight - weight_reduction * d_lg_weight_d_mtow)
                    / lg_weight ** 2.0
                )

                d_weight_reduction_factor_d_lg_height = (
                    weight_reduction
                    * (d_nlg_height_d_lg_height + d_mlg_height_d_lg_height)
                    / lg_weight ** 2.0
                )

                if wing_config == 3.0:
                    partials[
                        "data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MLW"
                    ] = 0.0
                    partials[
                        "data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MTOW"
                    ] = (1.08 * d_nlg_weight_d_mtow) * weight_reduction_factor + (
                        1.08 * nlg_weight
                    ) * d_weight_reduction_factor_d_mtow
                    partials[
                        "data:weight:airframe:landing_gear:front:mass",
                        "data:geometry:landing_gear:height",
                    ] = (1.08 * d_nlg_height_d_lg_height) * weight_reduction_factor + (
                        1.08 * nlg_weight
                    ) * d_weight_reduction_factor_d_lg_height
                else:
                    partials[
                        "data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MLW"
                    ] = 0.0
                    partials[
                        "data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MTOW"
                    ] = (
                        d_nlg_weight_d_mtow * weight_reduction_factor
                        + nlg_weight * d_weight_reduction_factor_d_mtow
                    )
                    partials[
                        "data:weight:airframe:landing_gear:front:mass",
                        "data:geometry:landing_gear:height",
                    ] = (
                        d_nlg_height_d_lg_height * weight_reduction_factor
                        + nlg_weight * d_weight_reduction_factor_d_lg_height
                    )
            else:
                if wing_config == 3.0:
                    partials[
                        "data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MLW"
                    ] = 0.0
                    partials[
                        "data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MTOW"
                    ] = (1.08 * d_nlg_weight_d_mtow)
                    partials[
                        "data:weight:airframe:landing_gear:front:mass",
                        "data:geometry:landing_gear:height",
                    ] = (
                        1.08 * d_nlg_height_d_lg_height
                    )
                else:
                    partials[
                        "data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MLW"
                    ] = 0.0
                    partials[
                        "data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MTOW"
                    ] = d_nlg_weight_d_mtow
                    partials[
                        "data:weight:airframe:landing_gear:front:mass",
                        "data:geometry:landing_gear:height",
                    ] = d_nlg_height_d_lg_height
        else:
            mlg_weight = (0.0117 - aircraft_type * 0.0012) * mlw ** 0.95 * lg_height ** 0.43
            nlg_weight = (
                (0.048 - aircraft_type * 0.008)
                * mlw ** 0.67
                * lg_height ** 0.43
                * (1.0 + 0.8 * carrier_based)
            )
            lg_weight = mlg_weight + nlg_weight

            d_nlg_weight_d_mlw = (
                -(
                    0.67
                    * lg_height ** 0.43
                    * (0.008 * aircraft_type - 0.048)
                    * (0.8 * carrier_based + 1.0)
                )
                / mlw ** 0.33
            )
            d_mlg_weight_d_mlw = (
                -(0.95 * lg_height ** 0.43 * (0.0012 * aircraft_type - 0.0117)) / mlw ** 0.05
            )

            d_nlg_height_d_lg_height = (
                -(
                    0.43
                    * mlw ** 0.67
                    * (0.008 * aircraft_type - 0.048)
                    * (0.8 * carrier_based + 1.0)
                )
                / lg_height ** 0.57
            )
            d_mlg_height_d_lg_height = (
                -(0.43 * mlw ** 0.95 * (0.0012 * aircraft_type - 0.0117)) / lg_height ** 0.57
            )

            if not is_retractable:
                weight_reduction = 1.4 * mtow / 100.0
                weight_reduction_factor = (mlg_weight + nlg_weight - weight_reduction) / (
                    mlg_weight + nlg_weight
                )

                d_weight_reduction_factor_d_mlw = (
                    weight_reduction * (d_nlg_weight_d_mlw + d_mlg_weight_d_mlw) / lg_weight ** 2.0
                )

                d_weight_reduction_d_mtow = -1.4 / (100.0 * lg_weight)

                d_weight_reduction_factor_d_lg_height = (
                    weight_reduction
                    * (d_nlg_height_d_lg_height + d_mlg_height_d_lg_height)
                    / lg_weight ** 2.0
                )

                if wing_config == 3.0:
                    partials[
                        "data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MLW"
                    ] = (1.08 * d_nlg_weight_d_mlw) * weight_reduction_factor + (
                        1.08 * nlg_weight
                    ) * d_weight_reduction_factor_d_mlw
                    partials[
                        "data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MTOW"
                    ] = (1.08 * nlg_weight) * d_weight_reduction_d_mtow
                    partials[
                        "data:weight:airframe:landing_gear:front:mass",
                        "data:geometry:landing_gear:height",
                    ] = (1.08 * d_nlg_height_d_lg_height) * weight_reduction_factor + (
                        1.08 * nlg_weight
                    ) * d_weight_reduction_factor_d_lg_height
                else:
                    partials[
                        "data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MLW"
                    ] = (
                        d_nlg_weight_d_mlw * weight_reduction_factor
                        + nlg_weight * d_weight_reduction_factor_d_mlw
                    )
                    partials[
                        "data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MTOW"
                    ] = (nlg_weight * d_weight_reduction_d_mtow)
                    partials[
                        "data:weight:airframe:landing_gear:front:mass",
                        "data:geometry:landing_gear:height",
                    ] = (
                        d_nlg_height_d_lg_height * weight_reduction_factor
                        + nlg_weight * d_weight_reduction_factor_d_lg_height
                    )
            else:
                if wing_config == 3.0:
                    partials[
                        "data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MLW"
                    ] = (1.08 * d_nlg_weight_d_mlw)
                    partials[
                        "data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MTOW"
                    ] = 0.0
                    partials[
                        "data:weight:airframe:landing_gear:front:mass",
                        "data:geometry:landing_gear:height",
                    ] = (
                        1.08 * d_nlg_height_d_lg_height
                    )
                else:
                    partials[
                        "data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MLW"
                    ] = d_nlg_weight_d_mlw
                    partials[
                        "data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MTOW"
                    ] = 0.0
                    partials[
                        "data:weight:airframe:landing_gear:front:mass",
                        "data:geometry:landing_gear:height",
                    ] = d_nlg_height_d_lg_height

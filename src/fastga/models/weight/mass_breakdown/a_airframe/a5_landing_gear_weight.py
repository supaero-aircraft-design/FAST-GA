"""
Python module for landing gear weight calculation, part of the airframe mass computation.
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

from .constants import SERVICE_LANDING_GEAR_MASS, SUBMODEL_LANDING_GEAR_MASS_LEGACY

oad.RegisterSubmodel.active_models[SERVICE_LANDING_GEAR_MASS] = SUBMODEL_LANDING_GEAR_MASS_LEGACY


@oad.RegisterSubmodel(SERVICE_LANDING_GEAR_MASS, SUBMODEL_LANDING_GEAR_MASS_LEGACY)
class ComputeLandingGearWeight(om.ExplicitComponent):
    """
    Weight estimation for landing gears

    Based on a statistical analysis. See :cite:`wells:2017` for the formula and
    :cite:`raymer:2012` for the weight reduction factor for non retractable landing gears
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="lb")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:geometry:landing_gear:height", val=np.nan, units="inch")

        self.add_input("data:geometry:landing_gear:type", val=np.nan)
        self.add_input("data:geometry:wing_configuration", val=np.nan)

        self.add_output("data:weight:airframe:landing_gear:main:mass", units="lb")
        self.add_output("data:weight:airframe:landing_gear:front:mass", units="lb")

        self.declare_partials(
            "*",
            ["data:geometry:landing_gear:type", "data:geometry:wing_configuration"],
            method="fd",
        )

        self.declare_partials(
            "*",
            [
                "data:weight:aircraft:MLW",
                "data:weight:aircraft:MTOW",
                "data:geometry:landing_gear:height",
            ],
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mlw = inputs["data:weight:aircraft:MLW"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        lg_height = inputs["data:geometry:landing_gear:height"]
        is_retractable = inputs["data:geometry:landing_gear:type"]
        wing_config = inputs["data:geometry:wing_configuration"]

        carrier_based = 0.0
        aircraft_type = 0.0  # One for fighter/attack aircraft

        # TODO : In the future updates the type of landing gear could be an option which would make
        #  the computation of that partials way easier.

        mlg_weight = (0.0117 - aircraft_type * 0.0012) * mlw**0.95 * lg_height**0.43
        nlg_weight = (
            (0.048 - aircraft_type * 0.008)
            * mlw**0.67
            * lg_height**0.43
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
            wing_config_const = 1.08
        else:
            wing_config_const = 1.0

        outputs["data:weight:airframe:landing_gear:main:mass"] = (
            mlg_weight * weight_reduction_factor * wing_config_const
        )
        outputs["data:weight:airframe:landing_gear:front:mass"] = (
            nlg_weight * weight_reduction_factor * wing_config_const
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mlw = inputs["data:weight:aircraft:MLW"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        lg_height = inputs["data:geometry:landing_gear:height"]
        is_retractable = inputs["data:geometry:landing_gear:type"]
        wing_config = inputs["data:geometry:wing_configuration"]

        carrier_based = 0.0
        aircraft_type = 0.0  # One for fighter/attack aircraft

        mlg_weight = (0.0117 - aircraft_type * 0.0012) * mlw**0.95 * lg_height**0.43
        nlg_weight = (
            (0.048 - aircraft_type * 0.008)
            * mlw**0.67
            * lg_height**0.43
            * (1.0 + 0.8 * carrier_based)
        )

        if wing_config == 3.0:
            wing_config_const = 1.08
        else:
            wing_config_const = 1.0

        if not is_retractable:
            weight_reduction = 1.4 * mtow / 100.0
            weight_reduction_factor = (mlg_weight + nlg_weight - weight_reduction) / (
                mlg_weight + nlg_weight
            )

            partials[
                "data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MTOW"
            ] = -nlg_weight * wing_config_const * (mlg_weight + nlg_weight) ** -1 * 1.4 / 100.0
            partials["data:weight:airframe:landing_gear:main:mass", "data:weight:aircraft:MTOW"] = (
                -mlg_weight * wing_config_const * (mlg_weight + nlg_weight) ** -1 * 1.4 / 100.0
            )
            partials["data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MLW"] = (
                wing_config_const
                * (
                    0.67
                    * (0.048 - aircraft_type * 0.008)
                    * mlw ** (-0.33)
                    * lg_height**0.43
                    * (1.0 + 0.8 * carrier_based)
                    * weight_reduction_factor
                    + nlg_weight
                    * weight_reduction
                    * (nlg_weight + mlg_weight) ** (-2)
                    * (
                        (0.0117 - aircraft_type * 0.0012) * 0.95 * mlw ** (-0.05) * lg_height**0.43
                        + (0.048 - aircraft_type * 0.008)
                        * 0.67
                        * mlw ** (-0.33)
                        * lg_height**0.43
                        * (1.0 + 0.8 * carrier_based)
                    )
                )
            )
            partials["data:weight:airframe:landing_gear:main:mass", "data:weight:aircraft:MLW"] = (
                wing_config_const
                * (
                    0.95
                    * (0.0117 - aircraft_type * 0.0012)
                    * mlw ** (-0.05)
                    * lg_height**0.43
                    * weight_reduction_factor
                    + mlg_weight
                    * weight_reduction
                    * (nlg_weight + mlg_weight) ** (-2)
                    * (
                        (0.0117 - aircraft_type * 0.0012) * 0.95 * mlw ** (-0.05) * lg_height**0.43
                        + (0.048 - aircraft_type * 0.008)
                        * 0.67
                        * mlw ** (-0.33)
                        * lg_height**0.43
                        * (1.0 + 0.8 * carrier_based)
                    )
                )
            )

            partials[
                "data:weight:airframe:landing_gear:front:mass", "data:geometry:landing_gear:height"
            ] = wing_config_const * (
                0.43
                * (0.048 - aircraft_type * 0.008)
                * mlw**0.67
                * lg_height ** (-0.57)
                * (1.0 + 0.8 * carrier_based)
                * weight_reduction_factor
                + nlg_weight
                * weight_reduction
                * (nlg_weight + mlg_weight) ** (-2)
                * (
                    (0.0117 - aircraft_type * 0.0012) * mlw**0.95
                    + (0.048 - aircraft_type * 0.008) * mlw**0.67 * (1.0 + 0.8 * carrier_based)
                )
                * 0.43
                * lg_height ** (-0.57)
            )

            partials[
                "data:weight:airframe:landing_gear:main:mass", "data:geometry:landing_gear:height"
            ] = wing_config_const * (
                (0.0117 - aircraft_type * 0.0012)
                * mlw**0.95
                * 0.43
                * lg_height ** (-0.57)
                * weight_reduction_factor
                + mlg_weight
                * weight_reduction
                * (nlg_weight + mlg_weight) ** (-2)
                * (
                    (0.0117 - aircraft_type * 0.0012) * mlw**0.95
                    + (0.048 - aircraft_type * 0.008) * mlw**0.67 * (1.0 + 0.8 * carrier_based)
                )
                * 0.43
                * lg_height ** (-0.57)
            )

        else:
            partials[
                "data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MTOW"
            ] = 0.0

            partials["data:weight:airframe:landing_gear:main:mass", "data:weight:aircraft:MTOW"] = (
                0.0
            )

            partials["data:weight:airframe:landing_gear:front:mass", "data:weight:aircraft:MLW"] = (
                wing_config_const
                * 0.67
                * (0.048 - aircraft_type * 0.008)
                * mlw ** (-0.33)
                * lg_height**0.43
                * (1.0 + 0.8 * carrier_based)
            )

            partials["data:weight:airframe:landing_gear:main:mass", "data:weight:aircraft:MLW"] = (
                wing_config_const
                * 0.95
                * (0.0117 - aircraft_type * 0.0012)
                * mlw ** (-0.05)
                * lg_height**0.43
            )

            partials[
                "data:weight:airframe:landing_gear:front:mass", "data:geometry:landing_gear:height"
            ] = wing_config_const * (
                0.43
                * (0.048 - aircraft_type * 0.008)
                * mlw**0.67
                * lg_height ** (-0.57)
                * (1.0 + 0.8 * carrier_based)
            )

            partials[
                "data:weight:airframe:landing_gear:main:mass", "data:geometry:landing_gear:height"
            ] = wing_config_const * (
                (0.0117 - aircraft_type * 0.0012) * mlw**0.95 * 0.43 * lg_height ** (-0.57)
            )

"""
Estimation of fuel lines weight.
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

import warnings

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent

import fastoad.api as oad

from .constants import SUBMODEL_FUEL_SYSTEM_MASS

oad.RegisterSubmodel.active_models[
    SUBMODEL_FUEL_SYSTEM_MASS
] = "fastga.submodel.weight.mass.propulsion.fuel_system.legacy"


@oad.RegisterSubmodel(
    SUBMODEL_FUEL_SYSTEM_MASS, "fastga.submodel.weight.mass.propulsion.fuel_system.legacy"
)
class ComputeFuelLinesWeight(ExplicitComponent):
    """
    Weight estimation for fuel lines

    Based on a statistical analysis. See :cite:`raymer:2012` but can also be found in
    :cite:`gudmundsson:2013`
    """

    def setup(self):

        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)
        self.add_input("data:weight:aircraft:MFW", val=np.nan, units="lb")
        self.add_input("data:propulsion:fuel_type", val=np.nan)

        self.add_output("data:weight:propulsion:fuel_lines:mass", units="lb")

        self.declare_partials(
            "data:weight:propulsion:fuel_lines:mass", "data:weight:aircraft:MFW", method="fd"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        tank_nb = 2.0  # Number of fuel tanks is assumed to be two, 1 per semi-wing
        engine_nb = inputs["data:geometry:propulsion:engine:count"]
        fuel_mass = inputs["data:weight:aircraft:MFW"]
        fuel_type = inputs["data:propulsion:fuel_type"]

        # The 0.5**0.363 refers to the ratio between the total fuel quantity and the total fuel
        # quantity plus the quantity in integral tanks. We will assume that we only have integral
        # tank hence the 0.5

        if fuel_type == 1.0:
            m_vol_fuel = 718.9  # gasoline volume-mass [kg/m**3], cold worst case, Avgas
        elif fuel_type == 2.0:
            m_vol_fuel = 860.0  # Diesel volume-mass [kg/m**3], cold worst case
        elif fuel_type == 3.0:
            m_vol_fuel = 804.0  # Jet-A1 volume mass [kg/m**3], cold worst case
        else:
            m_vol_fuel = 718.9
            warnings.warn("Fuel type {} does not exist, replaced by type 1!".format(fuel_type))

        k_fsp = m_vol_fuel * 0.008345
        # In lbs/gal

        b2 = (
            2.49
            * (fuel_mass / k_fsp) ** 0.726
            * 0.5 ** 0.363
            * tank_nb ** 0.242
            * engine_nb ** 0.157
        )  # mass formula in lb

        outputs["data:weight:propulsion:fuel_lines:mass"] = b2


@oad.RegisterSubmodel(
    SUBMODEL_FUEL_SYSTEM_MASS, "fastga.submodel.weight.mass.propulsion.fuel_system.flops"
)
class ComputeFuelLinesWeightFLOPS(ExplicitComponent):
    """
    Weight estimation for fuel lines

    Based on a statistical analysis. See :cite:`wells:2017`
    """

    def setup(self):

        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)
        self.add_input("data:weight:aircraft:MFW", val=np.nan, units="lb")

        self.add_output("data:weight:propulsion:fuel_lines:mass", units="lb")

        self.declare_partials(
            of="data:weight:propulsion:fuel_lines:mass",
            wrt=[
                "data:weight:aircraft:MFW",
                "data:geometry:propulsion:engine:count",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        engine_nb = inputs["data:geometry:propulsion:engine:count"]
        fuel_mass = inputs["data:weight:aircraft:MFW"]

        b2 = 1.07 * fuel_mass ** 0.58 * engine_nb ** 0.43

        outputs["data:weight:propulsion:fuel_lines:mass"] = b2

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        engine_nb = inputs["data:geometry:propulsion:engine:count"]
        fuel_mass = inputs["data:weight:aircraft:MFW"]

        partials["data:weight:propulsion:fuel_lines:mass", "data:weight:aircraft:MFW"] = (
            1.07 * 0.58 * fuel_mass ** -0.42 * engine_nb ** 0.43
        )
        partials[
            "data:weight:propulsion:fuel_lines:mass", "data:geometry:propulsion:engine:count"
        ] = (1.07 * 0.43 * fuel_mass ** 0.58 * engine_nb ** -0.57)

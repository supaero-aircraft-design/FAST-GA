"""
Estimation of electric engine and associated component weight
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

import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader

# from fastga.models.propulsion.hybrid_propulsion.base import HybridEngineSet


class ComputeEEngineWeight(ExplicitComponent):
    """
    Computes the electric engine weight as the sum of the weights of motor, power electronics, cables and the propeller.
    Based on FAST-GA-ELEC.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):

        # self.add_input("data:geometry:propulsion:count", val=np.nan)
        self.add_input("data:weight:hybrid_powertrain:motor:mass", val=np.nan, units="kg")
        self.add_input("data:weight:hybrid_powertrain:power_electronics:mass", val=np.nan, units="kg")
        self.add_input("data:weight:hybrid_powertrain:cable:mass", val=np.nan, units="kg")
        self.add_input("data:weight:hybrid_powertrain:propeller:mass", val=np.nan, units="kg")

        self.add_output("data:weight:hybrid_powertrain:engine:mass", units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_mass = inputs["data:weight:hybrid_powertrain:motor:mass"]
        pe_mass = inputs["data:weight:hybrid_powertrain:power_electronics:mass"]
        cable_mass = inputs["data:weight:hybrid_powertrain:cable:mass"]
        prop_mass = inputs["data:weight:hybrid_powertrain:propeller:mass"]

        b4 = motor_mass + pe_mass + cable_mass + prop_mass

        outputs["data:weight:hybrid_powertrain:engine:mass"] = b4

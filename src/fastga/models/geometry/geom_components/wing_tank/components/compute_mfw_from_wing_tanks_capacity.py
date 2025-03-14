"""
Python module for MFW from tank volume computation class(es), part of the advanced MFW computation
method.
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

import warnings
import numpy as np
import openmdao.api as om


class ComputeMFWFromWingTanksCapacity(om.ExplicitComponent):
    """Computes the MFW from the capacity of the two wing tanks inside the aircraft wings."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m_vol_fuel = None

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input(
            "data:geometry:propulsion:tank:capacity",
            units="m**3",
            val=np.nan,
            desc="Capacity of both tanks on the aircraft",
        )
        self.add_input("data:propulsion:fuel_type", val=np.nan)

        self.add_output("data:weight:aircraft:MFW", units="kg", val=500.0)

        self.declare_partials(of="*", wrt="data:geometry:propulsion:tank:capacity", method="exact")
        self.declare_partials("*", "data:propulsion:fuel_type", method="fd")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_type = inputs["data:propulsion:fuel_type"]
        tank_capacity = inputs["data:geometry:propulsion:tank:capacity"]

        if fuel_type == 1.0:
            self.m_vol_fuel = 718.9  # gasoline volume-mass [kg/m**3], cold worst case, Avgas
        elif fuel_type == 2.0:
            self.m_vol_fuel = 860.0  # Diesel volume-mass [kg/m**3], cold worst case
        elif fuel_type == 3.0:
            self.m_vol_fuel = 804.0  # Jet-A1 volume mass [kg/m**3], cold worst case
        else:
            self.m_vol_fuel = 718.9
            warnings.warn(f"Fuel type {fuel_type} does not exist, replaced by type 1!")

        outputs["data:weight:aircraft:MFW"] = tank_capacity * self.m_vol_fuel

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["data:weight:aircraft:MFW", "data:geometry:propulsion:tank:capacity"] = (
            self.m_vol_fuel
        )

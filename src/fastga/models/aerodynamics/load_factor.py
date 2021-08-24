"""
    FAST - Copyright (c) 2016 ONERA ISAE
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
from .components.compute_vn import DOMAIN_PTS_NB

from openmdao.core.group import Group
from openmdao.core.explicitcomponent import ExplicitComponent

from .components.compute_vn import ComputeVNAndVH

from fastoad.module_management.service_registry import RegisterOpenMDAOSystem
from fastoad.module_management.constants import ModelDomain


@RegisterOpenMDAOSystem("fastga.aerodynamics.load_factor", domain=ModelDomain.AERODYNAMICS)
class LoadFactor(Group):
    """
    Models for computing the loads and characteristic speed and load factor of the aircraft
    """

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self.add_subsystem(
            "vn_diagram",
            ComputeVNAndVH(propulsion_id=self.options["propulsion_id"]),
            promotes=["*"],
        )
        self.add_subsystem("sizing_load_factor", _LoadFactorIdentification(), promotes=["*"])
        self.add_subsystem(
            "characteristic_speeds", _CharacteristicSpeedIdentification(), promotes=["*"]
        )


class _LoadFactorIdentification(ExplicitComponent):
    def setup(self):
        nan_array = np.full(DOMAIN_PTS_NB, np.nan)
        self.add_input("data:flight_domain:load_factor", val=nan_array, shape=DOMAIN_PTS_NB)

        self.add_output("data:mission:sizing:cs23:sizing_factor_ultimate")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        load_factor_array = inputs["data:flight_domain:load_factor"]
        outputs["data:mission:sizing:cs23:sizing_factor_ultimate"] = 1.5 * max(
            abs(max(load_factor_array)), abs(min(load_factor_array))
        )


class _CharacteristicSpeedIdentification(ExplicitComponent):
    def setup(self):
        nan_array = np.full(DOMAIN_PTS_NB, np.nan)
        self.add_input(
            "data:flight_domain:velocity", val=nan_array, shape=DOMAIN_PTS_NB, units="m/s"
        )

        self.add_output("data:flight_domain:diving_speed", units="m/s")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        velocity_array = inputs["data:flight_domain:velocity"]
        outputs["data:flight_domain:diving_speed"] = velocity_array[9]

"""FAST - Copyright (c) 2016 ONERA ISAE."""

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

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.group import Group

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

from .components.compute_vn import ComputeVNAndVH, DOMAIN_PTS_NB


@oad.RegisterOpenMDAOSystem("fastga.aerodynamics.load_factor", domain=ModelDomain.AERODYNAMICS)
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


class _LoadFactorIdentification(ExplicitComponent):
    def setup(self):
        nan_array = np.full(DOMAIN_PTS_NB, np.nan)
        self.add_input(
            "data:mission:sizing:cs23:flight_domain:mtow:velocity",
            val=nan_array,
            units="m/s",
            shape=DOMAIN_PTS_NB,
        )
        self.add_input(
            "data:mission:sizing:cs23:flight_domain:mtow:load_factor",
            val=nan_array,
            shape=DOMAIN_PTS_NB,
        )

        self.add_input(
            "data:mission:sizing:cs23:flight_domain:mzfw:load_factor",
            val=nan_array,
            shape=DOMAIN_PTS_NB,
        )

        self.add_input("data:mission:sizing:cs23:safety_factor", val=np.nan)

        self.add_output("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft")
        self.add_output("data:mission:sizing:cs23:sizing_factor:ultimate_mtow:positive")
        self.add_output("data:mission:sizing:cs23:sizing_factor:ultimate_mtow:negative")
        self.add_output("data:mission:sizing:cs23:sizing_factor:ultimate_mzfw:positive")
        self.add_output("data:mission:sizing:cs23:sizing_factor:ultimate_mzfw:negative")

        self.add_output("data:mission:sizing:cs23:characteristic_speed:va", units="m/s")
        self.add_output("data:mission:sizing:cs23:characteristic_speed:vc", units="m/s")
        self.add_output("data:mission:sizing:cs23:characteristic_speed:vd", units="m/s")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        safety_factor = inputs["data:mission:sizing:cs23:safety_factor"]

        load_factor_array_mtow = inputs["data:mission:sizing:cs23:flight_domain:mtow:load_factor"]
        load_factor_array_mzfw = inputs["data:mission:sizing:cs23:flight_domain:mzfw:load_factor"]
        velocity_array_mtow = inputs["data:mission:sizing:cs23:flight_domain:mtow:velocity"]

        ultimate_load_factor_mtow_pos = safety_factor * max(load_factor_array_mtow)
        ultimate_load_factor_mtow_neg = safety_factor * min(load_factor_array_mtow)
        ultimate_load_factor_mzfw_pos = safety_factor * max(load_factor_array_mzfw)
        ultimate_load_factor_mzfw_neg = safety_factor * min(load_factor_array_mzfw)

        outputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"] = max(
            ultimate_load_factor_mtow_pos,
            ultimate_load_factor_mzfw_pos,
            abs(ultimate_load_factor_mtow_neg),
            abs(ultimate_load_factor_mzfw_neg),
        )
        outputs[
            "data:mission:sizing:cs23:sizing_factor:ultimate_mtow:positive"
        ] = ultimate_load_factor_mtow_pos
        outputs[
            "data:mission:sizing:cs23:sizing_factor:ultimate_mtow:negative"
        ] = ultimate_load_factor_mtow_neg
        outputs[
            "data:mission:sizing:cs23:sizing_factor:ultimate_mzfw:positive"
        ] = ultimate_load_factor_mzfw_pos
        outputs[
            "data:mission:sizing:cs23:sizing_factor:ultimate_mzfw:negative"
        ] = ultimate_load_factor_mzfw_neg

        outputs["data:mission:sizing:cs23:characteristic_speed:va"] = max(
            velocity_array_mtow[2], velocity_array_mtow[4]
        )
        outputs["data:mission:sizing:cs23:characteristic_speed:vc"] = velocity_array_mtow[6]
        outputs["data:mission:sizing:cs23:characteristic_speed:vd"] = velocity_array_mtow[9]

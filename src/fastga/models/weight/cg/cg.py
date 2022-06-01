"""
FAST - Copyright (c) 2016 ONERA ISAE.
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

import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_CENTER_OF_GRAVITY

from .cg_components.constants import (
    SUBMODEL_PAYLOAD_CG,
    SUBMODEL_AIRCRAFT_CG_EXTREME,
    SUBMODEL_TANK_CG,
)


@oad.RegisterSubmodel(SUBMODEL_CENTER_OF_GRAVITY, "fastga.submodel.weight.cg.legacy")
class CG(om.Group):
    """Model that computes the global center of gravity."""

    def __init__(self, **kwargs):
        """Defining solvers for cg computation resolution."""
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.linear_solver = om.LinearBlockGS()

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self.add_subsystem(
            "tank_cg", oad.RegisterSubmodel.get_submodel(SUBMODEL_TANK_CG), promotes=["*"]
        )
        self.add_subsystem(
            "payload_cg", oad.RegisterSubmodel.get_submodel(SUBMODEL_PAYLOAD_CG), promotes=["*"]
        )
        propulsion_option = {"propulsion_id": self.options["propulsion_id"]}
        self.add_subsystem(
            "compute_cg",
            oad.RegisterSubmodel.get_submodel(
                SUBMODEL_AIRCRAFT_CG_EXTREME, options=propulsion_option
            ),
            promotes=["*"],
        )

        # Solvers setup
        self.nonlinear_solver.options["debug_print"] = True
        self.nonlinear_solver.options["err_on_non_converge"] = True
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 50

        self.linear_solver.options["err_on_non_converge"] = True
        self.linear_solver.options["iprint"] = 0
        self.linear_solver.options["maxiter"] = 10

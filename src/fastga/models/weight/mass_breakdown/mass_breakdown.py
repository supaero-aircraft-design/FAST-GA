"""
Main components for mass breakdown.
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

from .constants import (
    SUBMODEL_OWE,
    SUBMODEL_MLW,
    SUBMODEL_ZFW,
    SUBMODEL_MZFW,
    SUBMODEL_PAYLOAD_MASS,
)
from ..constants import SUBMODEL_MASS_BREAKDOWN
from fastga.models.options import PAYLOAD_FROM_NPAX


@oad.RegisterSubmodel(SUBMODEL_MASS_BREAKDOWN, "fastga.submodel.weight.mass_breakdown.legacy")
class MassBreakdown(om.Group):
    """
    Computes analytically the mass of each part of the aircraft, and the resulting sum,
    the Overall Weight Empty (OWE).

    Some models depend on MZFW (Max Zero Fuel Weight) and MTOW (Max TakeOff Weight),
    which depend on OWE.

    This model cycles for having consistent OWE, MZFW and MTOW based on MFW.

    Options:
    - payload_from_npax: If True (default), payload masses will be computed from NPAX, if False
                         design payload mass and maximum payload mass must be provided.
    """

    def __init__(self, **kwargs):
        """Defining solvers for mass breakdown computation resolution."""
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.linear_solver = om.LinearBlockGS()

    def initialize(self):
        self.options.declare(PAYLOAD_FROM_NPAX, types=bool, default=True)
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        if self.options[PAYLOAD_FROM_NPAX]:
            self.add_subsystem(
                "payload",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_PAYLOAD_MASS),
                promotes=["*"],
            )
        propulsion_option = {"propulsion_id": self.options["propulsion_id"]}
        self.add_subsystem(
            "owe",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_OWE, options=propulsion_option),
            promotes=["*"],
        )
        self.add_subsystem(
            "update_mlw", oad.RegisterSubmodel.get_submodel(SUBMODEL_MLW), promotes=["*"]
        )
        self.add_subsystem(
            "update_zfw", oad.RegisterSubmodel.get_submodel(SUBMODEL_ZFW), promotes=["*"]
        )
        self.add_subsystem(
            "update_mzfw", oad.RegisterSubmodel.get_submodel(SUBMODEL_MZFW), promotes=["*"]
        )

        # Solvers setup
        self.nonlinear_solver.options["debug_print"] = True
        self.nonlinear_solver.options["err_on_non_converge"] = True
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 50

        self.linear_solver.options["err_on_non_converge"] = True
        self.linear_solver.options["iprint"] = 0
        self.linear_solver.options["maxiter"] = 10

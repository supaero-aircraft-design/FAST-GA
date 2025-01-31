"""
Python module for wing weight calculation using the analytical breakdown.
Based on the model presented by Raquel ALONSO in her MAE research project report.
Reference:
Alonso Castilla, Raquel, et al.
"Wing structural model for overall aircraft design of distributed electric propulsion general aviation and regional aircraft."
Aerospace 9.1 (2021): 5.
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

import fastoad.api as oad
import openmdao.api as om

from .constants import SERVICE_WING_MASS, SUBMODEL_WING_MASS_ANALYTICAL
from .wing_components.compute_lower_flange import ComputeLowerFlange
from .wing_components.compute_misc_mass import ComputeMiscMass
from .wing_components.compute_primary_mass import ComputePrimaryMass
from .wing_components.compute_ribs_mass import ComputeRibsMass
from .wing_components.compute_secondary_mass import ComputeSecondaryMass
from .wing_components.compute_skin_mass import ComputeSkinMass
from .wing_components.compute_upper_flange import ComputeUpperFlange
from .wing_components.compute_web_mass import ComputeWebMass
from .wing_components.update_wing_mass import UpdateWingMass


@oad.RegisterSubmodel(SERVICE_WING_MASS, SUBMODEL_WING_MASS_ANALYTICAL)
class ComputeWingMassAnalytical(om.Group):
    """
    Computes analytically the  mass of each component of the wing and add them to get total wing
    mass

    Loop on the wing mass cause its both a relief force and the result.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.linear_solver = om.LinearBlockGS()

    def setup(self):
        self.add_subsystem("compute_web_mass_max_fuel", ComputeWebMass(), promotes=["*"])
        self.add_subsystem(
            "compute_web_mass_min_fuel", ComputeWebMass(min_fuel_in_wing=True), promotes=["*"]
        )
        self.add_subsystem("compute_upp_flange_mass_max_fuel", ComputeUpperFlange(), promotes=["*"])
        self.add_subsystem(
            "compute_upp_flange_mass_min_fuel",
            ComputeUpperFlange(min_fuel_in_wing=True),
            promotes=["*"],
        )
        self.add_subsystem("compute_low_flange_mass_max_fuel", ComputeLowerFlange(), promotes=["*"])
        self.add_subsystem(
            "compute_low_flange_mass_min_fuel",
            ComputeLowerFlange(min_fuel_in_wing=True),
            promotes=["*"],
        )
        self.add_subsystem("compute_skin_mass", ComputeSkinMass(), promotes=["*"])
        self.add_subsystem("compute_ribs_mass", ComputeRibsMass(), promotes=["*"])
        self.add_subsystem("compute_misc_mass", ComputeMiscMass(), promotes=["*"])
        self.add_subsystem("compute_primary_structure", ComputePrimaryMass(), promotes=["*"])
        self.add_subsystem("compute_secondary_structure", ComputeSecondaryMass(), promotes=["*"])
        self.add_subsystem("update_wing_mass", UpdateWingMass(), promotes=["*"])

        # Solver configuration
        self.nonlinear_solver.options["debug_print"] = True
        # self.nonlinear_solver.options["err_on_non_converge"] = True
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 100
        # self.nonlinear_solver.options["reraise_child_analysiserror"] = True
        self.nonlinear_solver.options["rtol"] = 1e-4

        # self.linear_solver.options["err_on_non_converge"] = True
        self.linear_solver.options["iprint"] = 0
        self.linear_solver.options["maxiter"] = 10
        self.linear_solver.options["rtol"] = 1e-4

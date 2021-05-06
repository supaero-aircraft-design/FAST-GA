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
import openmdao.api as om

from .cg_components.a_airframe import ComputeWingCG, ComputeFuselageCG, ComputeTailCG, ComputeFlightControlCG, \
    ComputeLandingGearCG
from .cg_components.b_propulsion import ComputeEngineCG, ComputeFuelLinesCG, ComputeTankCG
from .cg_components.c_systems import ComputePowerSystemsCG, ComputeLifeSupportCG, ComputeNavigationSystemsCG
from .cg_components.d_furniture import ComputePassengerSeatsCG
from .cg_components.payload import ComputePayloadCG
from .cg_components.global_cg import ComputeGlobalCG
from .cg_components.update_mlg import UpdateMLG


class CG(om.Group):
    """ Model that computes the global center of gravity """

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):

        self.add_subsystem("wing_cg", ComputeWingCG(), promotes=["*"])
        self.add_subsystem("fuselage_cg", ComputeFuselageCG(), promotes=["*"])
        self.add_subsystem("tail_cg", ComputeTailCG(), promotes=["*"])
        self.add_subsystem("flight_control_cg", ComputeFlightControlCG(), promotes=["*"])
        self.add_subsystem("landing_gear_cg", ComputeLandingGearCG(), promotes=["*"])
        self.add_subsystem("engine_cg", ComputeEngineCG(), promotes=["*"])
        self.add_subsystem("fuel_lines_cg", ComputeTankCG(), promotes=["*"])
        self.add_subsystem("tank_cg", ComputeFuelLinesCG(), promotes=["*"])
        self.add_subsystem("power_systems_cg", ComputePowerSystemsCG(), promotes=["*"])
        self.add_subsystem("life_support_cg", ComputeLifeSupportCG(), promotes=["*"])
        self.add_subsystem("navigation_systems_cg", ComputeNavigationSystemsCG(), promotes=["*"])
        self.add_subsystem("passenger_seats_cg", ComputePassengerSeatsCG(), promotes=["*"])
        self.add_subsystem("payload_cg", ComputePayloadCG(), promotes=["*"])
        self.add_subsystem("compute_cg", ComputeGlobalCG(propulsion_id=self.options['propulsion_id']), promotes=["*"])
        self.add_subsystem("update_mlg", UpdateMLG(), promotes=["*"])
        self.add_subsystem("aircraft", ComputeAircraftCG(), promotes=["*"])

        # Solvers setup
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options["debug_print"] = True
        self.nonlinear_solver.options["err_on_non_converge"] = True
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 50
        # self.nonlinear_solver.options["reraise_child_analysiserror"] = True
        # self.nonlinear_solver.options["rtol"] = 1e-5

        self.linear_solver = om.LinearBlockGS()
        self.linear_solver.options["err_on_non_converge"] = True
        self.linear_solver.options["iprint"] = 0
        self.linear_solver.options["maxiter"] = 10
        # self.linear_solver.options["rtol"] = 1e-5


class ComputeAircraftCG(om.ExplicitComponent):
    """ Compute position of aircraft CG from CG ratio """

    def setup(self):
    
        self.add_input("data:weight:aircraft:CG:aft:MAC_position", val=np.nan)
        self.add_input("data:weight:aircraft:CG:fwd:MAC_position", val=np.nan)
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")

        self.add_output("data:weight:aircraft:CG:aft:x", units="m")
        self.add_output("data:weight:aircraft:CG:fwd:x", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
    
        cg_aft_ratio = inputs["data:weight:aircraft:CG:aft:MAC_position"]
        cg_fwd_ratio = inputs["data:weight:aircraft:CG:fwd:MAC_position"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        mac_position = inputs["data:geometry:wing:MAC:at25percent:x"]

        outputs["data:weight:aircraft:CG:aft:x"] = (
            mac_position - 0.25 * l0_wing + cg_aft_ratio * l0_wing
        )
        # Comment this line if ComputeGlobalCG is used
        outputs["data:weight:aircraft:CG:fwd:x"] = (
            mac_position - 0.25 * l0_wing + cg_fwd_ratio * l0_wing
        )

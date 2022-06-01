"""Simple module for complete mission."""
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

import logging

import numpy as np
import openmdao.api as om

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

from fastga.models.performances.mission.takeoff import TakeOffPhase


from fastga.models.weight.cg.cg_variation import InFlightCGVariation
from .constants import (
    SUBMODEL_TAXI,
    SUBMODEL_CLIMB_SPEED,
    SUBMODEL_CLIMB,
    SUBMODEL_CRUISE,
    SUBMODEL_DESCENT,
    SUBMODEL_DESCENT_SPEED,
    SUBMODEL_RESERVES,
)

MAX_CALCULATION_TIME = 15  # time in seconds

_LOGGER = logging.getLogger(__name__)


@oad.RegisterOpenMDAOSystem("fastga.performances.mission", domain=ModelDomain.PERFORMANCE)
class Mission(om.Group):
    """
    Computes analytically the fuel mass necessary for each part of the flight cycle.

    Loop on the distance crossed during descent and cruise distance/fuel mass.

    """

    def __init__(self, **kwargs):
        """Defining solvers for mission computation resolution."""
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.linear_solver = om.LinearBlockGS()

    def initialize(self):
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)
        self.options.declare("out_file", default="", types=str)

    def setup(self):
        self.add_subsystem("in_flight_cg_variation", InFlightCGVariation(), promotes=["*"])
        taxi_out_options = {"taxi_out": True, "propulsion_id": self.options["propulsion_id"]}
        self.add_subsystem(
            "taxi_out",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_TAXI, options=taxi_out_options),
            promotes=["*"],
        )
        self.add_subsystem(
            "takeoff", TakeOffPhase(propulsion_id=self.options["propulsion_id"]), promotes=["*"]
        )
        options_mission = {
            "propulsion_id": self.options["propulsion_id"],
            "out_file": self.options["out_file"],
        }
        self.add_subsystem(
            "climb_speed",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CLIMB_SPEED),
            promotes=["*"],
        )
        self.add_subsystem(
            "climb",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CLIMB, options=options_mission),
            promotes=["*"],
        )
        self.add_subsystem(
            "cruise",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CRUISE, options=options_mission),
            promotes=["*"],
        )
        self.add_subsystem(
            "reserve", oad.RegisterSubmodel.get_submodel(SUBMODEL_RESERVES), promotes=["*"]
        )
        self.add_subsystem(
            "descent_speed",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_DESCENT_SPEED),
            promotes=["*"],
        )
        self.add_subsystem(
            "descent",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_DESCENT, options=options_mission),
            promotes=["*"],
        )
        taxi_out_options = {"taxi_out": False, "propulsion_id": self.options["propulsion_id"]}
        self.add_subsystem(
            "taxi_in",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_TAXI, options=taxi_out_options),
            promotes=["*"],
        )
        self.add_subsystem("update_fw", UpdateFW(), promotes=["*"])

        # Solvers setup
        self.nonlinear_solver.options["debug_print"] = True
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 100
        self.nonlinear_solver.options["rtol"] = 1e-2

        # self.linear_solver.options["err_on_non_converge"] = True
        self.linear_solver.options["iprint"] = 0
        self.linear_solver.options["maxiter"] = 10
        self.linear_solver.options["rtol"] = 1e-2


class UpdateFW(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:mission:sizing:taxi_out:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:cruise:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:reserve:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:descent:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_in:fuel", np.nan, units="kg")

        self.add_output("data:mission:sizing:fuel", val=0.0, units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        m_taxi_out = inputs["data:mission:sizing:taxi_out:fuel"]
        m_takeoff = inputs["data:mission:sizing:takeoff:fuel"]
        m_initial_climb = inputs["data:mission:sizing:initial_climb:fuel"]
        m_climb = inputs["data:mission:sizing:main_route:climb:fuel"]
        m_cruise = inputs["data:mission:sizing:main_route:cruise:fuel"]
        m_reserve = inputs["data:mission:sizing:main_route:reserve:fuel"]
        m_descent = inputs["data:mission:sizing:main_route:descent:fuel"]
        m_taxi_in = inputs["data:mission:sizing:taxi_in:fuel"]

        m_total = (
            m_taxi_out
            + m_takeoff
            + m_initial_climb
            + m_climb
            + m_cruise
            + m_reserve
            + m_descent
            + m_taxi_in
        )

        outputs["data:mission:sizing:fuel"] = m_total

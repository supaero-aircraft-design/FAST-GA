"""Estimation of payload range diagram points."""

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

import fastoad.api as oad
import numpy as np
import openmdao.api as om
from fastoad.module_management.constants import ModelDomain
from fastoad.openmdao.problem import AutoUnitsDefaultGroup

from fastga.command import api as api_cs23
from fastga.models.performances.mission.mission import Mission
from fastga.utils.options_checkers import check_propulsion_id

_LOGGER = logging.getLogger(__name__)


@oad.RegisterOpenMDAOSystem("fastga.performances.payload_range", domain=ModelDomain.PERFORMANCE)
class ComputePayloadRange(om.ExplicitComponent):
    """
    Payload Range. The minimal payload which defines point E is taken as two pilots. This class
    uses a blank xml file for the execution of the mission class. All the input quantities of the
    mission are created in a dict. generate_block_analysis still needs a xml file to be processed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fuel_problem = None

    def initialize(self):
        self.options.declare("propulsion_id", check_valid=check_propulsion_id)

    def setup(self):
        variables = api_cs23.list_variables(Mission(propulsion_id=self.options["propulsion_id"]))

        inputs_mission = [var for var in variables if var.is_input]

        for comp_input in inputs_mission:
            self.add_input(
                comp_input.name,
                val=np.nan,
                units=comp_input.metadata["units"],
                shape=comp_input.metadata["shape"],
            )
        self.add_input("data:weight:aircraft:max_payload", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:fuel", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MFW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MZFW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:OWE", val=np.nan, units="kg")

        self.add_output("data:payload_range:payload_array", units="kg", shape=5)
        self.add_output("data:payload_range:range_array", units="NM", shape=5)
        self.add_output("data:payload_range:specific_range_array", units="NM/kg", shape=5)

        self.declare_partials("*", "*", method="fd")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        payload_mission = inputs["data:weight:aircraft:payload"][0]
        max_payload = inputs["data:weight:aircraft:max_payload"][0]
        range_mission = inputs["data:TLAR:range"]
        fuel_mission = inputs["data:mission:sizing:fuel"]
        mfw = inputs["data:weight:aircraft:MFW"]
        mzfw = inputs["data:weight:aircraft:MZFW"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        owe = inputs["data:weight:aircraft:OWE"]
        mass_pilot = inputs["settings:weight:aircraft:payload:design_mass_per_passenger"]

        payload_array = []
        range_array = []
        sr_array = []

        # First setup the problem
        self.setup_fuel_problem(inputs=inputs, prop_id=self.options["propulsion_id"])

        # Point A : 0 fuel, max payload and mass < MTOW
        payload_array.append(max_payload)
        range_array.append(0)
        sr_array.append(0)

        # Point B : max payload, enough fuel to have mass = MTOW
        fuel_target_b = mtow - mzfw
        range_b = self.get_range(fuel_target_b, mtow, initial_range_guess=range_mission / 2)

        payload_array.append(max_payload)
        range_array.append(range_b[0])
        sr_array.append(range_b[0] / fuel_target_b[0])

        fuel_target_c = fuel_mission
        range_c = self.get_range(fuel_target_c, mtow)

        payload_array.append(payload_mission)
        range_array.append(range_c[0])
        sr_array.append(range_c[0] / fuel_target_c[0])

        # Point D : max fuel (MFW), enough payload to have mass = MTOW
        fuel_target_d = mfw
        payload_d = max_payload - (mfw - fuel_target_b)

        range_d = self.get_range(fuel_target_d, mtow)

        if payload_d < 2 * mass_pilot:
            _LOGGER.warning(
                "Point D computed but the payload for this point is lower than minimal payload (2 "
                "pilots) "
            )
        payload_array.append(payload_d[0])
        range_array.append(range_d[0])
        sr_array.append(range_d[0] / fuel_target_d[0])

        # Point E : max fuel (MFW), min payload and the aircraft resulting mass
        fuel_target_e = mfw
        payload_e = 0.0
        mass_aircraft = owe + mfw + payload_e
        range_e = self.get_range(fuel_target_e, mass_aircraft)

        payload_array.append(payload_e)
        range_array.append(range_e[0])
        sr_array.append(range_e[0] / fuel_target_e[0])

        # Conversion in nautical miles
        range_array = [i / 1852 for i in range_array]
        sr_array = [i / 1852 for i in sr_array]

        outputs["data:payload_range:payload_array"] = payload_array
        outputs["data:payload_range:range_array"] = range_array
        outputs["data:payload_range:specific_range_array"] = sr_array

    def setup_fuel_problem(self, inputs, prop_id):

        mission_component = AutoUnitsDefaultGroup()
        mission_component.add_subsystem(
            "system",
            Mission(propulsion_id=prop_id),
            promotes=["*"],
        )
        variables = api_cs23.list_variables(mission_component)
        var_inputs = [var.name for var in variables if var.is_input]
        var_units = [var.units for var in variables if var.is_input]
        var_shapes = [np.shape(var.val) for var in variables if var.is_input]

        input_zip = zip(var_inputs, var_units, var_shapes)

        # These should never change so we'll set them up like this
        ivc = om.IndepVarComp()
        for var_names, var_unit, var_shape in input_zip:
            if var_names not in {"data:TLAR:range", "data:weight:aircraft:MTOW"}:
                ivc.add_output(
                    name=var_names, val=inputs[var_names], units=var_unit, shape=var_shape
                )

        self.fuel_problem = oad.FASTOADProblem()
        model = self.fuel_problem.model

        model.add_subsystem("ivc", ivc, promotes_outputs=["*"])
        model.add_subsystem("mission", Mission(propulsion_id=prop_id), promotes=["*"])
        model.add_subsystem("distance_to_target", DistanceToTarget(), promotes=["*"])

        model.nonlinear_solver = om.NonlinearBlockGS()
        model.nonlinear_solver.options["iprint"] = 0
        model.nonlinear_solver.options["maxiter"] = 10
        model.nonlinear_solver.options["rtol"] = 1e-3
        model.nonlinear_solver.options["atol"] = 1

        model.linear_solver = om.LinearBlockGS()
        model.linear_solver.options["iprint"] = 0
        model.linear_solver.options["maxiter"] = 10
        model.linear_solver.options["rtol"] = 1e-3

        self.fuel_problem.setup()

    def get_range(self, fuel_target, mass, initial_range_guess=None):
        """
        Runs the cached OpenMDAO problems for the target fuel and aircraft mass and finds the
        corresponding range.

        :param fuel_target: target fuel to achieve
        :param mass: aircraft mass
        :param initial_range_guess: initial guess for the appropriate range, in m. If nothing is
        provided, the value at which the problem last converged will be used

        :return: the range which leads to the inputs fuel mass at input weight.
        """
        self.fuel_problem.set_val(name="data:weight:aircraft:MTOW", val=mass, units="kg")
        self.fuel_problem.set_val(
            name="data:mission:sizing:fuel_target", val=fuel_target, units="kg"
        )

        if initial_range_guess:
            self.fuel_problem.set_val("data:TLAR:range", val=initial_range_guess, units="m")

        self.fuel_problem.run_model()

        return self.fuel_problem.get_val(name="data:TLAR:range", units="m")


class DistanceToTarget(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:mission:sizing:fuel", units="kg", val=np.nan)
        self.add_input("data:mission:sizing:fuel_target", units="kg", val=np.nan)
        self.add_input("data:mission:sizing:main_route:climb:distance", units="m", val=np.nan)
        self.add_input("data:mission:sizing:main_route:descent:distance", units="m", val=np.nan)
        self.add_input("data:mission:sizing:main_route:cruise:distance", units="m", val=np.nan)

        self.add_output("data:TLAR:range", 370400, units="m")  # 200 nm in m

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        climb_range = inputs["data:mission:sizing:main_route:climb:distance"]
        descent_range = inputs["data:mission:sizing:main_route:descent:distance"]
        cruise_range = inputs["data:mission:sizing:main_route:cruise:distance"]

        current_fuel = inputs["data:mission:sizing:fuel"]
        target_fuel = inputs["data:mission:sizing:fuel_target"]

        outputs["data:TLAR:range"] = (
            (climb_range + cruise_range + descent_range) * target_fuel / current_fuel
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        climb_range = inputs["data:mission:sizing:main_route:climb:distance"]
        descent_range = inputs["data:mission:sizing:main_route:descent:distance"]
        cruise_range = inputs["data:mission:sizing:main_route:cruise:distance"]

        current_fuel = inputs["data:mission:sizing:fuel"]
        target_fuel = inputs["data:mission:sizing:fuel_target"]

        partials["data:TLAR:range", "data:mission:sizing:main_route:climb:distance"] = (
            target_fuel / current_fuel
        )
        partials["data:TLAR:range", "data:mission:sizing:main_route:descent:distance"] = (
            target_fuel / current_fuel
        )
        partials["data:TLAR:range", "data:mission:sizing:main_route:cruise:distance"] = (
            target_fuel / current_fuel
        )

        partials["data:TLAR:range", "data:mission:sizing:fuel"] = (
            -(climb_range + cruise_range + descent_range) * target_fuel / current_fuel**2.0
        )
        partials["data:TLAR:range", "data:mission:sizing:fuel_target"] = (
            climb_range + cruise_range + descent_range
        ) / current_fuel

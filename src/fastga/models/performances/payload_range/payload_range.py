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
import numpy as np
import openmdao.api as om


from scipy.optimize import fsolve

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain
from fastoad.openmdao.problem import AutoUnitsDefaultGroup

from fastga.command import api as api_cs23
from fastga.models.performances.mission.mission import Mission

_LOGGER = logging.getLogger(__name__)


@oad.RegisterOpenMDAOSystem("fastga.performances.payload_range", domain=ModelDomain.PERFORMANCE)
class ComputePayloadRange(om.ExplicitComponent):
    """
    Payload Range. The minimal payload which defines point E is taken as two pilots. This class
    uses a blank xml file for the execution of the mission class. All the input quantities of the
    mission are created in a dict. generate_block_analysis still needs a xml file to be processed.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

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

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        payload_mission = inputs["data:weight:aircraft:payload"]
        max_payload = inputs["data:weight:aircraft:max_payload"]
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

        # Point A : 0 fuel, max payload and mass < MTOW
        payload_array.append(max_payload)
        range_array.append(0)
        sr_array.append(0)

        # Point B : max payload, enough fuel to have mass = MTOW
        fuel_target_b = mtow - mzfw
        range_b, _, ier, message = fsolve(
            self.fuel_function,
            range_mission / 2,
            args=(fuel_target_b, mtow, inputs, self.options["propulsion_id"]),
            xtol=0.01,
            full_output=True,
        )
        if ier != 1:
            _LOGGER.warning("Computation of point B failed. Error message : %s", message)

        payload_array.append(max_payload)
        range_array.append(range_b[0])
        sr_array.append(range_b[0] / fuel_target_b)

        fuel_target_c = fuel_mission
        range_c, _, ier, message = fsolve(
            self.fuel_function,
            range_mission / 2,
            args=(fuel_target_c, mtow, inputs, self.options["propulsion_id"]),
            xtol=0.01,
            full_output=True,
        )
        if ier != 1:
            _LOGGER.warning("Computation of point C failed. Error message : %s", message)

        payload_array.append(payload_mission)
        range_array.append(range_c)
        sr_array.append(range_c / fuel_target_c)

        # Point D : max fuel (MFW), enough payload to have mass = MTOW
        fuel_target_d = mfw
        payload_d = max_payload - (mfw - fuel_target_b)

        range_d, _, ier, message = fsolve(
            self.fuel_function,
            range_mission,
            args=(fuel_target_d, mtow, inputs, self.options["propulsion_id"]),
            xtol=0.01,
            full_output=True,
        )
        if ier != 1:
            _LOGGER.warning("Computation of point D failed. Error message : %s", message)

        if payload_d < 2 * mass_pilot:
            _LOGGER.warning(
                "Point D computed but the payload for this point is lower than minimal payload (2 "
                "pilots) "
            )
        payload_array.append(payload_d)
        range_array.append(range_d[0])
        sr_array.append(range_d[0] / fuel_target_d)

        # Point E : max fuel (MFW), min payload and the aircraft resulting mass
        fuel_target_e = mfw
        payload_e = 0.0
        mass_aircraft = owe + mfw + payload_e
        range_e, _, ier, message = fsolve(
            self.fuel_function,
            range_mission,
            args=(fuel_target_e, mass_aircraft, inputs, self.options["propulsion_id"]),
            xtol=0.01,
            full_output=True,
        )
        if ier != 1:
            _LOGGER.warning("Computation of point E failed. Error message : %s", message)

        payload_array.append(payload_e)
        range_array.append(range_e[0])
        sr_array.append(range_e[0] / fuel_target_e)

        # Conversion in nautical miles
        range_array = [i / 1852 for i in range_array]
        sr_array = [i / 1852 for i in sr_array]

        outputs["data:payload_range:payload_array"] = payload_array
        outputs["data:payload_range:range_array"] = range_array
        outputs["data:payload_range:specific_range_array"] = sr_array

    @staticmethod
    def fuel_function(range_parameter, fuel_target, mass, inputs, prop_id):

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

        ivc = om.IndepVarComp()
        for var_names, var_unit, var_shape in input_zip:
            if var_names != "data:TLAR:range" and var_names != "data:weight:aircraft:MTOW":
                ivc.add_output(
                    name=var_names, val=inputs[var_names], units=var_unit, shape=var_shape
                )

        ivc.add_output(name="data:TLAR:range", val=range_parameter, units="m")
        ivc.add_output(name="data:weight:aircraft:MTOW", val=mass, units="kg")

        problem = oad.FASTOADProblem()
        model = problem.model

        model.add_subsystem("ivc", ivc, promotes_outputs=["*"])
        model.add_subsystem("mission", Mission(propulsion_id=prop_id), promotes=["*"])

        model.nonlinear_solver = om.NonlinearBlockGS()
        model.nonlinear_solver.options["iprint"] = 0
        model.nonlinear_solver.options["maxiter"] = 10
        model.nonlinear_solver.options["rtol"] = 1e-3

        model.linear_solver = om.LinearBlockGS()
        model.linear_solver.options["iprint"] = 0
        model.linear_solver.options["maxiter"] = 10
        model.linear_solver.options["rtol"] = 1e-3

        problem.setup()

        problem.run_model()

        fuel = problem.get_val("data:mission:sizing:fuel", units="kg")

        return fuel - fuel_target

"""
    Estimation of fuselage aiframe mass
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
from fastoad.module_management._bundle_loader import BundleLoader
from fastga.command import api as api_cs23
import os.path as pth
import os
import logging
from scipy.optimize import fsolve

from fastga.models.performances.mission import Mission
from fastoad.io import VariableIO
from fastga.models.propulsion.fuel_propulsion.base import FuelEngineSet

current_path: str = r"/src/fastga/models/performances/payload_range"
XML_FILE = pth.join(current_path, 'blank.xml')

_LOGGER = logging.getLogger(__name__)

class ComputePayloadRange(om.ExplicitComponent):
    """
        Payload Range.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):

        variables = api_cs23.list_variables(Mission(propulsion_id="fastga.wrapper.propulsion.basicIC_engine"))

        inputs_mission = [var for var in variables if var.is_input]

        for input in inputs_mission:
            self.add_input(input.name, val=np.nan, units=input.metadata["units"], shape=input.metadata["shape"])
        self.add_input("data:weight:aircraft:max_payload", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:fuel", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MFW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MZFW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:OWE", val=np.nan, units="kg")

        self.add_output("data:payload_range:payload_array", units="kg", shape=5)
        self.add_output("data:payload_range:range_array", units="m", shape=5)
        self.add_output("data:payload_range:specific_range_array", units="m/kg", shape=5)

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
        range_b, dic, _, _ = fsolve(self.fuel_function, range_mission / 2, args=(fuel_target_b, mtow, inputs), xtol=1, full_output= True)

        payload_array.append(max_payload)
        range_array.append(range_b[0])
        sr_array.append(range_b[0] / fuel_target_b)

        _LOGGER.warning("Number of fsolve iterations for point B :")
        _LOGGER.warning(dic["nfev"])

        # Point C : design point
        payload_array.append(payload_mission)
        range_array.append(range_mission)
        sr_array.append(range_mission / fuel_mission)

        # Point D : max fuel (MFW), enough payload to have mass = MTOW
        fuel_target_d = mfw
        payload_d = max_payload - (mfw - fuel_target_b)

        if payload_d > 2 * mass_pilot:
            range_d, dic, _, _ = fsolve(self.fuel_function, range_mission, args=(fuel_target_d, mtow, inputs), xtol=1, full_output=True)
            _LOGGER.warning("Number of fsolve iterations for point D :")
            _LOGGER.warning(dic["nfev"])
        else:
            range_d = 0
            _LOGGER.warning("Point D not computed, payload for this point lower than minimal payload")
        payload_array.append(payload_d)
        range_array.append(range_d[0])
        sr_array.append(range_d[0] / fuel_target_d)

        # Point E : max fuel (MFW), min payload (2 pilots) and the aircraft resulting mass
        fuel_target_e = mfw
        payload_e = 2 * mass_pilot
        mass_aircraft = owe + mfw + payload_e
        range_e, dic, _, _ = fsolve(self.fuel_function, range_mission, args=(fuel_target_e, mass_aircraft, inputs), xtol=1, full_output=True)

        _LOGGER.warning("Number of fsolve iterations for point E :")
        _LOGGER.warning(dic["nfev"])

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
    def fuel_function(range_parameter, fuel_target, mass, inputs):

        variables = api_cs23.list_variables(Mission(propulsion_id="fastga.wrapper.propulsion.basicIC_engine"))
        inputs_mission = [var for var in variables if var.is_input]
        for i in range(len(inputs_mission)):
            inputs_mission[i].metadata["value"] = inputs[inputs_mission[i].name]

        var_names = np.array([var.name for var in inputs_mission])
        var_value = np.array([var.metadata["value"].tolist() for var in inputs_mission], dtype='object')
        var_units = np.array([var.metadata["units"] for var in inputs_mission])

        index_range = np.where(var_names == "data:TLAR:range")
        var_value[index_range[0]] = range_parameter

        index_mtow = np.where(var_names == "data:weight:aircraft:MTOW")
        var_value[index_mtow[0]] = mass

        compute_fuel = api_cs23.generate_block_analysis(
            Mission(propulsion_id="fastga.wrapper.propulsion.basicIC_engine"),
            var_names.tolist(),
            XML_FILE,
            True,
        )

        inputs_dict = {}

        for i in range(np.size(var_names)):
            inputs_dict.update({var_names[i]: (var_value[i], var_units[i])})

        output = compute_fuel(inputs_dict)

        variables = VariableIO(XML_FILE).read()
        fuel = variables["data:mission:sizing:fuel"].value

        return fuel - fuel_target

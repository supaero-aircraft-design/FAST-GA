""" Module that sizes the battery in a hybrid propulsion model (FC-B configuration). """

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

import openmdao.api as om
from .resources import fuel_cell, battery
import numpy as np
import math as math
import matplotlib.pyplot as plt


class ComputeBatteries(om.ExplicitComponent):
    """
    Based on Zhao, Tianyuan, "Propulsive Battery Packs Sizing for Aviation Applications" (2018). Dissertations and
    Theses. 393. (https://commons.erau.edu/edt/393)
    """

    def setup(self):
        self.add_input("data:propulsion:hybrid_powertrain:battery:type", val=0, units=None,
                       desc=" Optional type of battery - 0 is None and 1 is default Li-ion cells")
        self.add_input("data:geometry:hybrid_powertrain:battery:nb_packs", val=np.nan, units=None)
        self.add_input("data:propulsion:hybrid_powertrain:battery:input_current", val=np.nan, units="A")
        self.add_input("data:geometry:hybrid_powertrain:battery:cell_diameter", val=np.nan, units="m")
        self.add_input("data:geometry:hybrid_powertrain:battery:cell_length", val=np.nan, units="m")
        self.add_input("data:geometry:hybrid_powertrain:battery:cell_mass", val=np.nan, units="kg")
        self.add_input("data:propulsion:hybrid_powertrain:battery:cell_capacity", val=np.nan, units="A*h")
        self.add_input("data:propulsion:hybrid_powertrain:battery:cell_nom_voltage", val=np.nan, units="V")
        self.add_input("data:propulsion:hybrid_powertrain:battery:max_C_rate", val=np.nan, units="h**-1")
        self.add_input("data:propulsion:hybrid_powertrain:battery:int_resistance", val=np.nan, units="ohm")
        self.add_input("data:propulsion:hybrid_powertrain:battery:SOC", val=np.nan, units=None)
        # self.add_input("data:propulsion:hybrid_powertrain:battery:current_limit", val=np.nan, units="A")
        # self.add_input("data:propulsion:hybrid_powertrain:battery:cutoff_voltage", val=np.nan, units="V")
        self.add_input("data:propulsion:hybrid_powertrain:battery:sys_nom_voltage", val=np.nan, units="V")
        self.add_input("data:propulsion:hybrid_powertrain:motor:efficiency", val=np.nan, units=None)
        self.add_input("data:propulsion:hybrid_powertrain:motor:TO_power", val=np.nan, units="W")
        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:design_power", val=np.nan, units='W')
        self.add_input("data:mission:sizing:takeoff:duration", val=np.nan, units='s')
        self.add_input("data:mission:sizing:main_route:climb:battery_energy", val=np.nan, units='W*h')
        self.add_input("data:mission:sizing:main_route:descent:battery_energy", val=np.nan, units='W*h')
        self.add_input("data:mission:sizing:main_route:reserve:battery_energy", val=np.nan, units='W*h')

        self.add_output("data:geometry:hybrid_powertrain:battery:N_series", units=None)
        self.add_output("data:geometry:hybrid_powertrain:battery:N_parallel", units=None)
        self.add_output("data:geometry:hybrid_powertrain:battery:pack_volume", units='m**3')
        self.add_output("data:geometry:hybrid_powertrain:battery:tot_volume", units='m**3')

        self.declare_partials('*', '*', method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        battery_type = inputs['data:propulsion:hybrid_powertrain:battery:type']
        nb_packs = inputs['data:geometry:hybrid_powertrain:battery:nb_packs']
        input_current = inputs['data:propulsion:hybrid_powertrain:battery:input_current']
        cell_d = inputs['data:geometry:hybrid_powertrain:battery:cell_diameter']
        cell_l = inputs['data:geometry:hybrid_powertrain:battery:cell_length']
        cell_c = inputs['data:propulsion:hybrid_powertrain:battery:cell_capacity']
        cell_m = inputs['data:geometry:hybrid_powertrain:battery:cell_mass']
        cell_nom_V = inputs['data:propulsion:hybrid_powertrain:battery:cell_nom_voltage']
        max_C_rate = inputs['data:propulsion:hybrid_powertrain:battery:max_C_rate']
        int_res = inputs['data:propulsion:hybrid_powertrain:battery:int_resistance']
        SOC = inputs['data:propulsion:hybrid_powertrain:battery:SOC']
        # current_limit = inputs['data:propulsion:hybrid_powertrain:battery:current_limit']
        # cutoff_V = inputs['data:propulsion:hybrid_powertrain:battery:cutoff_voltage']
        nom_V = inputs['data:propulsion:hybrid_powertrain:battery:sys_nom_voltage']
        motor_eff = inputs['data:propulsion:hybrid_powertrain:motor:efficiency']
        TO_power = inputs['data:propulsion:hybrid_powertrain:motor:TO_power']
        fc_power = inputs['data:propulsion:hybrid_powertrain:fuel_cell:design_power']
        TO_time = inputs['data:mission:sizing:takeoff:duration']
        climb_energy = inputs['data:mission:sizing:main_route:climb:battery_energy']
        descent_energy = inputs['data:mission:sizing:main_route:descent:battery_energy']
        reserve_energy = inputs['data:mission:sizing:main_route:reserve:battery_energy']

        batt = battery.Battery(battery_type=battery_type,
                               in_current=input_current,
                               cell_diameter=cell_d,
                               cell_length=cell_l,
                               cell_capacity=cell_c,
                               cell_mass=cell_m,
                               cell_nom_volt=cell_nom_V,
                               max_C_rate=max_C_rate,
                               int_resistance=int_res,
                               TO_time=TO_time,
                               descent_energy = descent_energy,
                               climb_energy=climb_energy,
                               reserve_energy=reserve_energy,
                               SOC=SOC,
                               # current_limit=current_limit,
                               # cutoff_voltage=cutoff_V,
                               sys_nom_voltage=nom_V,
                               motor_TO_power=TO_power,
                               motor_eff=motor_eff,
                               fc_power=fc_power)

        N_series = batt.compute_nb_cells_ser()
        N_par = batt.compute_nb_cells_par()
        vol = batt.compute_pack_volume()
        tot_vol = nb_packs * vol

        outputs['data:geometry:hybrid_powertrain:battery:N_series'] = N_series
        outputs['data:geometry:hybrid_powertrain:battery:N_parallel'] = N_par
        outputs['data:geometry:hybrid_powertrain:battery:pack_volume'] = vol
        outputs['data:geometry:hybrid_powertrain:battery:tot_volume'] = tot_vol

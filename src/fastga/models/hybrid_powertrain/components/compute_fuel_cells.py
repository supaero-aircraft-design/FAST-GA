""" Module that computes the fuel cell in a hybrid propulsion model (FC-B configuration). """

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
from .resources.fuel_cell import FuelCell
import numpy as np


class ComputeFuelCells(om.ExplicitComponent):
    """
    PowerCellution V Stack (a PEM fuel cell supplied with hydrogen and air) is used as a reference for now :
    https://www.datocms-assets.com/36080/1611437781-v-stack.pdf.
    This propulsion model assumes that the fuel cell system should provide the entire cruise power ; battery power
    would only come into play during maneuvers, take-off, climbing and landing.
    Power requirements for all components other than the compressor are neglected.
    """

    def setup(self):
        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:user_defined_power", val=np.nan, units='W',
                       desc="User defined output FC power. Used to size the FC if the cruise power is lower.")
        self.add_input("data:propulsion:hybrid_powertrain:battery:sys_nom_voltage", val=np.nan, units='V')
        self.add_input("data:mission:sizing:main_route:cruise:power_fuel_cell", val=np.nan, units='W')
        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:stack_pressure", val=np.nan, units='Pa')
        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:nominal_pressure", val=np.nan, units='Pa')
        self.add_input("data:propulsion:hybrid_powertrain:compressor:power", val=0, units='W')
        self.add_input("data:geometry:hybrid_powertrain:fuel_cell:number_stacks", val=2,
                       desc = 'Used only in the determination of the geometry of the FC')
        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:fc_type", val=0, units=None,
                       desc="Optional type of fuel cell - 0 is None and 1 is default fuel cell")
        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:design_current_density", val=0.144, units='A/cm**2')

        self.add_output("data:propulsion:hybrid_powertrain:fuel_cell:stack_power", units='W')
        self.add_output("data:propulsion:hybrid_powertrain:fuel_cell:design_current", units='A')
        self.add_output("data:propulsion:hybrid_powertrain:fuel_cell:output_power", units="W")

        self.add_output("data:geometry:hybrid_powertrain:fuel_cell:number_cells", units=None,
                        desc="Total number of cells in the stack(s)")
        self.add_output("data:geometry:hybrid_powertrain:fuel_cell:stack_height", units='cm')
        self.add_output("data:geometry:hybrid_powertrain:fuel_cell:stack_volume", units='cm**3')
        self.add_output("data:propulsion:hybrid_powertrain:fuel_cell:cell_voltage", units='V')
        self.add_output("data:propulsion:hybrid_powertrain:fuel_cell:cooling_power", units='W')
        self.add_output("data:propulsion:hybrid_powertrain:fuel_cell:efficiency", units=None)
        self.add_output("data:propulsion:hybrid_powertrain:fuel_cell:ox_mass_flow", units="kg/s",
                        desc='oxygene mass flow rate for all stack')
        self.add_output('data:propulsion:hybrid_powertrain:fuel_cell:hyd_mass_flow', units="kg/s",
                        desc='hydrogene mass flow rate for all stack,'
                             'divide by stack_number to have per stack hydrogene consumption')
        self.add_output("data:geometry:hybrid_powertrain:fuel_cell:stack_area", units='cm**2')

        self.declare_partials('*', '*', method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        voltage_level = inputs['data:propulsion:hybrid_powertrain:battery:sys_nom_voltage']
        user_power = inputs['data:propulsion:hybrid_powertrain:fuel_cell:user_defined_power']
        compressor_power = inputs['data:propulsion:hybrid_powertrain:compressor:power']
        stack_pressure = inputs['data:propulsion:hybrid_powertrain:fuel_cell:stack_pressure']
        stack_current_density = inputs['data:propulsion:hybrid_powertrain:fuel_cell:design_current_density']
        nominal_pressure = inputs['data:propulsion:hybrid_powertrain:fuel_cell:nominal_pressure']
        fc_type = inputs['data:propulsion:hybrid_powertrain:fuel_cell:fc_type']
        nb_stacks = inputs['data:geometry:hybrid_powertrain:fuel_cell:number_stacks']
        cruise_power = inputs["data:mission:sizing:main_route:cruise:power_fuel_cell"]

        # Creating a FuelCell instance on which all computing methods will be called. This instance encapsulates the
        # computation of all the fuel cell stacks if there are more than 1.

        pow_per_fc = max(user_power, cruise_power) / nb_stacks
        fc = FuelCell(required_power=pow_per_fc,
                      compressor_power=compressor_power,
                      stack_pressure=stack_pressure,
                      current_density=stack_current_density,
                      voltage_level=voltage_level,
                      nom_pressure=nominal_pressure,
                      fc_type=fc_type)

        design_power = fc.compute_design_power()
        design_current = fc.compute_design_current()
        v_cell = fc.compute_cell_V()
        nb_cells = fc.compute_nb_cell()
        stack_height = fc.compute_fc_height(nb_cells)
        stack_volume = fc.compute_fc_volume(nb_cells)
        p_cooling = fc.compute_cooling_power()
        eff = fc.compute_efficiency()
        ox_flow = fc.compute_ox_mass_flow() * nb_stacks
        hyd_flow = fc.compute_hyd_mass_flow() * nb_stacks
        cell_area = fc.compute_cell_area()

        outputs['data:propulsion:hybrid_powertrain:fuel_cell:stack_power'] = design_power
        outputs['data:propulsion:hybrid_powertrain:fuel_cell:output_power'] = pow_per_fc * nb_stacks
        outputs['data:geometry:hybrid_powertrain:fuel_cell:number_cells'] = nb_cells
        outputs['data:geometry:hybrid_powertrain:fuel_cell:stack_height'] = stack_height
        outputs['data:geometry:hybrid_powertrain:fuel_cell:stack_volume'] = stack_volume
        outputs['data:propulsion:hybrid_powertrain:fuel_cell:cell_voltage'] = v_cell
        outputs['data:propulsion:hybrid_powertrain:fuel_cell:cooling_power'] = p_cooling
        outputs['data:propulsion:hybrid_powertrain:fuel_cell:efficiency'] = eff
        outputs['data:propulsion:hybrid_powertrain:fuel_cell:ox_mass_flow'] = ox_flow
        outputs['data:propulsion:hybrid_powertrain:fuel_cell:hyd_mass_flow'] = hyd_flow
        outputs['data:geometry:hybrid_powertrain:fuel_cell:stack_area'] = cell_area
        outputs['data:propulsion:hybrid_powertrain:fuel_cell:design_current'] = design_current

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
        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:design_current", val=np.nan, units='A')
        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:required_power", val=np.nan, units='W')
        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:stack_pressure", val=np.nan, units='Pa')
        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:nominal_pressure", val=np.nan, units='Pa')
        self.add_input("data:geometry:hybrid_powertrain:fuel_cell:stack_area", val=759.50, units='cm**2')
        self.add_input("data:propulsion:hybrid_powertrain:compressor:power", val=0, units='W')
        self.add_input("data:geometry:hybrid_powertrain:fuel_cell:number_stacks", val=2)
        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:fc_type", val=0, units=None,
                       desc="Optional type of fuel cell - 0 is None and 1 is default fuel cell")

        self.add_output("data:propulsion:hybrid_powertrain:fuel_cell:design_power", units='W')
        self.add_output("data:propulsion:hybrid_powertrain:fuel_cell:design_current_density", units='A/cm**2')
        self.add_output("data:geometry:hybrid_powertrain:fuel_cell:number_cells", units=None,
                        desc="Total number of cells in the stack(s)")
        self.add_output("data:geometry:hybrid_powertrain:fuel_cell:stack_height", units='cm')
        self.add_output("data:geometry:hybrid_powertrain:fuel_cell:stack_volume", units='cm**3')
        self.add_output("data:propulsion:hybrid_powertrain:fuel_cell:cell_voltage", units='V')
        self.add_output("data:propulsion:hybrid_powertrain:fuel_cell:cooling_power", units='W')
        self.add_output("data:propulsion:hybrid_powertrain:fuel_cell:efficiency", units=None)
        self.add_output("data:propulsion:hybrid_powertrain:fuel_cell:ox_mass_flow", units="kg/s")
        self.add_output('data:propulsion:hybrid_powertrain:fuel_cell:hyd_mass_flow', units="kg/s")

        self.declare_partials('*', '*', method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        design_current = inputs['data:propulsion:hybrid_powertrain:fuel_cell:design_current']
        required_power = inputs['data:propulsion:hybrid_powertrain:fuel_cell:required_power']
        compressor_power = inputs['data:propulsion:hybrid_powertrain:compressor:power']
        stack_pressure = inputs['data:propulsion:hybrid_powertrain:fuel_cell:stack_pressure']
        stack_area = inputs['data:geometry:hybrid_powertrain:fuel_cell:stack_area']
        nominal_pressure = inputs['data:propulsion:hybrid_powertrain:fuel_cell:nominal_pressure']
        fc_type = inputs['data:propulsion:hybrid_powertrain:fuel_cell:fc_type']
        nb_stacks = inputs['data:geometry:hybrid_powertrain:fuel_cell:number_stacks']

        # Creating a FuelCell instance on which all computing methods will be called. This instance encapsulates the
        # computation of all the fuel cell stacks if there are more than 1.

        fc = FuelCell(stack_current=design_current,
                      required_power=required_power,
                      compressor_power=compressor_power,
                      stack_pressure=stack_pressure,
                      stack_area=stack_area,
                      nom_pressure=nominal_pressure,
                      fc_type=fc_type,
                      number_stacks=nb_stacks)

        design_power = fc.compute_design_power()
        V_cell = fc.compute_cell_V()
        nb_cells = fc.compute_nb_cell()
        stack_height = fc.compute_fc_height(nb_cells)
        stack_volume = fc.compute_fc_volume(nb_cells)
        P_cooling = fc.compute_cooling_power()
        eff = fc.compute_ref_efficiency()
        ox_flow = fc.compute_ox_mass_flow()
        hyd_flow = fc.compute_hyd_mass_flow()
        current_density = fc.compute_design_current_density()

        outputs['data:propulsion:hybrid_powertrain:fuel_cell:design_power'] = design_power
        outputs['data:geometry:hybrid_powertrain:fuel_cell:number_cells'] = nb_cells
        outputs['data:geometry:hybrid_powertrain:fuel_cell:stack_height'] = stack_height
        outputs['data:geometry:hybrid_powertrain:fuel_cell:stack_volume'] = stack_volume
        outputs['data:propulsion:hybrid_powertrain:fuel_cell:cell_voltage'] = V_cell
        outputs['data:propulsion:hybrid_powertrain:fuel_cell:cooling_power'] = P_cooling
        outputs['data:propulsion:hybrid_powertrain:fuel_cell:efficiency'] = eff
        outputs['data:propulsion:hybrid_powertrain:fuel_cell:ox_mass_flow'] = ox_flow
        outputs['data:propulsion:hybrid_powertrain:fuel_cell:hyd_mass_flow'] = hyd_flow
        outputs['data:propulsion:hybrid_powertrain:fuel_cell:design_current_density'] = current_density

"""Module that contains all methods addressing the sizing of fuel cells."""
# -*- coding: utf-8 -*-
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
from scipy.interpolate import interp1d

# Update the dictionary so that is includes more specific values and maybe parameters such as stack area, cell area...
FuelCellTypes = {
    'POWERCELLUTION_V_STACK': {  # Reference fuel cell
        'HYD_STOICH_RATIO': 1.5,
        'OX_STOICH_RATIO': 2,
        'P_NOM': 10000,  # [Pa]
        'STACK_AREA': 759.50  # [cm**2]
    },
    # 'POWER_GENERATION_S30': {  # Based on the PowerCellution V Stack
    #     'HYD_STOIC_RATIO': 1.5,
    #     'OX_STOICH_RATIO': 2,
    #     'P_NOM': 10000,  # [Pa]
    #     'MAX_NET_POWER': 30,  # [kW]
    #     'STACK_AREA': 759.50  # [cm**2]
    # }
}


class FuelCell(object):
    """
    PowerCellution V Stack (a PEM fuel cell supplied with hydrogen and air) is used as a reference for now :
        https://www.datocms-assets.com/36080/1611437781-v-stack.pdf.
    By default, stack_number is set at 2 fuel cell stacks and reference parameters are set to those of the PowerCellution
    V Stack ; compressor power is also set at 0 if not specified.
    """

    def __init__(self,
                 required_power: float,
                 stack_pressure: float,
                 nom_pressure: float,
                 current_density: float,
                 voltage_level: float,
                 compressor_power: float = 0,
                 fc_type: int = 0,
                 stack_area: float = 1.0,
                 n_cells: int = 10):

        # If a type of FC has been specified, some parameters are set to those of the data in FuelCellTypes
        self.data = FuelCellTypes['POWERCELLUTION_V_STACK']  # Using data of the reference FC for now
        if fc_type == 0:
            self.nom_pressure = nom_pressure
            self.current_density = current_density
        else:  # Type '1.0'
            # Retrieving data from the dictionary
            # self.data = FuelCellTypes[f'{fc_type}']
            self.stack_area = self.data['STACK_AREA']
            self.nom_pressure = self.data['P_NOM']

        self.required_power = required_power
        self.compressor_power = compressor_power
        self.stack_pressure = stack_pressure
        self.voltage_level = voltage_level
        self.stack_area = stack_area
        self.cell_number = n_cells

    @staticmethod
    def compute_fc_weight(cell_number: int, cell_area: float):
        # Computes the weight of the fuel cell stack(s) given the total number of cells.
        # It is assumed that weight can be described as a linear function of the number of cells with the parameters :
        #     a = 0.1028153153153153
        #     b = 8.762162162162165
        # Those parameters are based on data retrieved from the PowerCellution V Stack fuel cell :
        #     https://www.datocms-assets.com/36080/1611437781-v-stack.pdf.
        # The fuel cell of PowerCellution V stack being a ref 759.5cm**2, a ratio of area is applied

        a_ratio = cell_area/759.5

        return (cell_number * 0.103 + 8.762)*a_ratio  # [kg]

    def compute_fc_height(self, cell_number: int):
        # Computes the height of a single fuel cell stack given the total number of cells and the number of stacks.
        # It is assumed that height can be described as a linear function of the number of cells of parameters :
        #     a = 1.3907657657657655
        #     b = 91.5081081081081
        # Those parameters are based on data retrieved from the PowerCellution V Stack fuel cell :
        #     https://www.datocms-assets.com/36080/1611437781-v-stack.pdf.

        return (cell_number * 1.39 + 91.51) / 10  # [cm]

    def compute_fc_volume(self, cell_number: int):
        # Computes the volume of a single fuel cell stack given the total number of cells and considering constructor data
        # for the reference fuel cell Power Cellution V Stack :
        #     https://www.datocms-assets.com/36080/1611437781-v-stack.pdf.

        vol = self.compute_cell_area() * self.compute_fc_height(cell_number)
        return vol  # [cm**3]

    def compute_design_power(self):
        # Computes the total required power of the fuel cell system.

        return self.compressor_power + self.required_power  # [W]

    def compute_design_current(self):

        design_pow = self.compute_design_power()

        return design_pow / self.voltage_level

    def compute_cell_area(self):

        design_current = self.compute_design_current()

        return design_current/self.current_density

    def compute_nb_cell(self):
        # Computes the number of cell needed considering design power of the FC system.

        # Determining one cell's design power
        P_cell_des = self.compute_design_current() * self.compute_design_cell_V()

        # Determining total required power
        tot_power = self.compute_design_power()

        # Returning the number of cells
        N_fc = np.ceil(tot_power / P_cell_des)
        return N_fc

    def compute_design_cell_V(self):

        # Using the design current density
        i = self.current_density

        return self.compute_cell_V(i)

    def compute_cell_V(self, i):

        """
        Computes the voltage in a cell considering the polarization curve model found here :
            https://repository.tudelft.nl/islandora/object/uuid%3A6e274095-9920-4d20-9e11-d5b76363e709
        First 3 parameters were adjusted to fit the i-V curve of the Power Cellution V Stack.
        + add a curve_fitting option to adjust to another type of fuel cell ?

        i : the cell current density in A/cm**2

        """

        # Defining the fitting parameters of the polarization curve - first 3 values are modified
        V0 = 0.64  # [V]
        B = 2.4 * 0.014  # [V/ln(A/cm**2)]
        R = 3 * 0.24  # [Ohm.cm**2]
        m = 5.63E-06  # [V]
        n = 11.42  # [cm**2/A]
        C = 0.05  # [V]

        # Returning cell design voltage
        V = V0 - B * np.log(i) - R * i - m * np.exp(n * i) + C * np.log(self.stack_pressure / self.nom_pressure)
        return V

    def compute_cooling_power(self):
        # Computes the cooling power needed considering required power and assuming that what is not delivered as
        # electricity is produced as heat.
        # Based on the work done in FAST-GA-AMPERE.

        P_cooling = self.compute_design_power() / self.compute_efficiency() - self.compute_design_power()-\
                    self.compressor_power
        return P_cooling

    def compute_hyd_mass_flow(self, i):

        """
        Computes the hydrogen mass flow rate required by the FC stack given required power and average cell voltage.
        Based on constructor data of the PowerCellution V Stack.

        i : current (A))
        """

        # Defining constants
        M_H2 = 2.016  # [g/mol]
        stoich_ratio = self.data['HYD_STOICH_RATIO']  # Hydrogen stoichiometric ratio for the chosen FC
        F = 96485  # [C/mol] - Faraday Constant

        # hyd_mass_flow = M_H2 * self.required_power * stoich_ratio / (2 * self.compute_design_cell_V() * F)  # [g/s]
        hyd_mass_flow = M_H2 * i / (2 * F)  # [g/s]
        return hyd_mass_flow / 1000  # [kg/s]

    def compute_design_hyd_mass_flow(self):

        i = self.compute_design_current()

        return self.compute_nb_cell() * self.compute_hyd_mass_flow(i)


    def compute_ox_mass_flow(self, i):

        """Computes the oxygen mass flow rate required by the FC stack given required power and average cell voltage.
        Based on constructor data of the PowerCellution V Stack.

        i: current (A)

        """

        # Defining constants
        M_O2 = 31.998  # [g/mol]
        stoich_ratio = self.data['OX_STOICH_RATIO']  # Oxygen stoichiometric ratio for the chosen FC
        F = 96485  # [C/mol] - Faraday Constant

        # ox_mass_flow = M_O2 * self.required_power * stoich_ratio / (4 * self.compute_design_cell_V() * F)  # [g/s]
        ox_mass_flow = M_O2 * i / (4 * F)  # [g/s]
        return ox_mass_flow / 1000  # [kg/s]

    def compute_design_ox_mass_flow(self):

        i=self.compute_design_current()

        return self.compute_nb_cell() * self.compute_ox_mass_flow(i)

    def compute_ref_efficiency(self):
        # Calculates the efficiency of the fuel cell system : calculations are based on reference data for the Power
        # Generation System 30 : https://www.datocms-assets.com/36080/1611437727-power-generation-system-30.pdf
        # Curve parameters were determined beforehand for this fuel cell.
        # Constructor data is provided for currents between 50 and 200 A so outside of these values, efficiency should not
        # be computed.

        # Defining net_power/I linear regression coefficients
        a = 0.136
        b = 2.786
        # Defining heat_power/I linear regression coefficients
        a1 = 0.181
        b1 = -2.943

        # Determining efficiency
        eff = (a * self.compute_design_current() + b) / ((a + a1) * self.compute_design_current() + b + b1)
        return eff

    def compute_efficiency(self):
        # Calculates the efficiency of the fuel cell system using a more generic formula.
        # Not used at the moment because provides very low values.

        # Defining constants
        H2_ed = 34100.0  # [Wh/kg] - Hydrogen energy density

        # Determining efficiency
        eff = self.compute_design_power() / (self.compute_design_hyd_mass_flow() * 3600.0 * H2_ed)
        return eff

    def get_hyd_flow(self, out_power):

        max_amps = (self.required_power+self.compressor_power) / self.voltage_level
        amps = np.linspace(1.0, float(max_amps)*1.2, 100)
        power_vs_i = self.compute_cell_V(amps/self.stack_area)*amps*self.cell_number
        i_vs_power = interp1d(power_vs_i, amps)
        working_current = i_vs_power(out_power+self.compressor_power)

        hyd_flow = self.compute_hyd_mass_flow(working_current)*self.cell_number

        import matplotlib.pyplot as plt

        # plt.figure()
        # plt.plot(amps, power_vs_i)
        # power = np.linspace(1.0,self.required_power+self.compressor_power, 50)
        # amps_1 = []
        # for i in range(len(power))
        #     amps_1.append(i_vs_power(power[i]))
        # plt.plot(amps_1, power, '+')

        # eff = out_power / (hyd_flow * 3600.0 * 34100)

        return hyd_flow
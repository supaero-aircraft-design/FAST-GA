"""Module that contains all methods addressing the sizing of the batteries."""

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

import math
import numpy as np
from .constants import CELL_WEIGHT_FRACTION

CellTypes = {
    'LG-HG2': {  # Type '1' in input
        # Common type of Li-ion battery - references found here : https://www.researchgate.net/publication/319935703_A_Fuel_Cell_System_Sizing_Tool_Based_on_Current_Production_Aircraft
        'DIAMETER': 18,  # [mm]
        'LENGTH': 65,  # [mm]
        'SPECIFIC_ENERGY': 240,  # [Wh/kg]
        'ENERGY_DENSITY': 670,  # [Wh/L]
        'RATED_CAPACITY': 3000,  # [mAh]
        'CELL_MASS': 44.5,  # [g]
        'I_MAX': 20,  # [A]
        'V_CUT_OFF': 2.5,  # [V]
        'V_NOM': 3.6  # [V]
    },
    'LI-S': {  # Type '2' in input
        # At the moment : not developed enough to serve general aviation propulsion purposes
        'SPECIFIC_ENERGY': 550,  # [Wh/kg] - 2023 assessment - 600 in 2030
        'ENERGY_DENSITY': 620,  # [Wh/L] - 2023 assessment - 700 in 2030
    }
}


class Battery(object):
    # Batteries are sized to provide additional power during take-off, climbing and landing phases. Other than that they
    # are designed for emergency backup if the fuel cell system were to fail, therefore to provide the same amount of
    # power as the fuel cell system for around 20~30 min to allow the plane to land safely from any altitude.
    # Assuming cylindrical battery cells and hexagonal stacking.
    # Li-ion battery cells are considered for now.
    # If there are more than one battery pack, we assume that :
    #     - there is a maximum of 2 batteries
    #     - the second battery pack serves as an emergency backup in case the first one fails : _init_ parameters define
    #     the sizing of a single battery pack
    # Based on :
    #     https://commons.erau.edu/cgi/viewcontent.cgi?article=1392&context=edt


    def __init__(
            self,
            cell_diameter: float,
            cell_length: float,
            cell_capacity: float,
            cell_nom_volt: float,
            cell_mass: float,
            max_C_rate: float,
            int_resistance: float,
            SOC_end_of_flight: float,
            current_limit: float,
            sys_nom_voltage: float,
            battery_type: int,
    ):

        if battery_type == 0:
            self.cell_d = cell_diameter
            self.cell_l = cell_length
            self.cell_c = cell_capacity
            self.cell_m = cell_mass
            self.cell_nom_V = cell_nom_volt
            self.i_max = current_limit
            # self.co_V = cutoff_voltage
        else:  # Type '1' is LG-HG2 cells
            self.data = CellTypes['LG-HG2']
            self.cell_d = self.data['DIAMETER'] * 1e-3  # [m]
            self.cell_l = self.data['LENGTH'] * 1e-3  # [m]
            self.cell_c = self.data['RATED_CAPACITY'] * 1e-3  # [Ah]
            self.cell_m = self.data['CELL_MASS'] * 1e-3  # [kg]
            self.cell_nom_V = self.data['V_NOM']  # [V]
            self.i_max = self.data['I_MAX']
            # self.co_V = data['V_CUT_OFF']

        self.max_C_rate = max_C_rate
        self.int_resistance = int_resistance
        self.SOC_end_of_flight = SOC_end_of_flight
        self.nom_voltage = sys_nom_voltage


    def compute_voltage(self, SOC):
        # Computes battery voltage considering cell voltage : for now cell voltage computation considers a
        # simplified method to compute voltage ('compute_voltage') instead of a more complex one
        # ('compute_V_cell_shepherd')

        return self.compute_V_cell(SOC) * self.compute_nb_cells_ser()

    def compute_V_cell(self, SOC):
        # Using a linear approximation between 500 mAh and 2750 mAh to compute cell voltage given State of Charge
        # (See https://commons.erau.edu/cgi/viewcontent.cgi?article=1392&context=edt)

        V0 = 4.14  # [V] - Battery cell voltage when battery at full capacity with a 0 discharging current
        V_soc = 0.94  # [V]

        # This function is used to size the battery, assumes max current is drawn.
        return V0 - V_soc * (1-SOC) - self.int_resistance * self.i_max

    def compute_nb_cells_ser(self):
        # Number of cells in series is computed considering nominal voltage of the battery system and
        # cell voltage at the end of the flight when maximum current is drawn.
        # Check conditions : Ns·VCellMin ≥ VMotorMin and Ns·VCellMax ≤ VMotorMax

        return math.ceil(self.nom_voltage / self.compute_V_cell(self.SOC_end_of_flight))

    def compute_nb_cells_par(self, total_energy, max_current):
        # Number of cells in parallel is sized for energy or for max current demand

        # Compute the battery energy using the SOC_end_of_flight
        battery_energy = total_energy / (1-self.SOC_end_of_flight)
        nb_cells_energy = math.ceil(battery_energy / (self.cell_c * self.nom_voltage))

        nb_cells_power = math.ceil(max_current / self.i_max)

        return max(nb_cells_power, nb_cells_energy)

    def compute_pack_volume(self, n_serie, n_par):
        # Computes the volume of a single battery pack.
        # Assuming :
        #     - hexagonal packing for our calculation (and using a formula found in sources specified above)
        #     - identical packs if there are more than 1

        BATT_OVERHEAD = 0.60  # Overhead factor - Considering 40% of the battery pack consists of overhead components
        eta = 0.907  # Hexagonal packing density

        V_pack = math.pi * self.cell_d ** 2 * self.cell_l * n_serie * n_par\
                 / (4 * eta * BATT_OVERHEAD)  # [m**3]
        return V_pack

'''
    def compute_required_power(self):
        # Required power is computed considering additional power needed during take-off
        return self.motor_TO_power / self.motor_eff - self.fc_power

    def compute_discharge_current(self):
        # Computing battery system output current
        return self.compute_required_power() / self.compute_voltage()
        
    def compute_capacity(self):
        # Computes battery system capacity - does not consider Sheferd model for cell voltage modelisation for now
        return self.compute_required_power() / self.nom_voltage
    
    def compute_V_cell_shepherd(self, time: float, current: float):
        # Considering Shepherd's empirical model for battery modelling. Equations and reference data can be found here :
        # https://repository.tudelft.nl/islandora/object/uuid%3A6e274095-9920-4d20-9e11-d5b76363e709
        # https://www.sciencedirect.com/science/article/pii/S0360319914031218
        # This method computes the voltage of a cell given its time in operation.
        # Not used for now but may be more accurate than 'compute_V_cell'.

        """
        :param time: total time of operation in s
        :param current: constant current drawn from the battery
        """

        # Defining constants - Considering nominal battery parameters
        V0 = self.cell_nom_V  # [V] - Nominal voltage
        K = 0.08726  # [1/Ah] - Polarization constant
        Q = self.compute_capacity()  # [Ah] - Maximum battery capacity
        R = self.int_resistance  # [Ohm] - Ohmic resistance
        A = 2.451  # [V] - Exponential voltage
        C = self.compute_discharge_current() / Q
        p0 = 3.057  # [mA/h]
        p1 = -0.6613  # [mA]
        p2 = 0.1273  # [mAh]
        p3 = -9.331E-5  # [mAh**2]
        B = p3 * C ** 3 + p2 * C ** 2 + p1 * C + p0  # [1/Ah] - Exponential capacity

        V = V0 - K * Q / (Q - current * time) + A * np.exp(-B * current * time) - R * current
        return V
'''
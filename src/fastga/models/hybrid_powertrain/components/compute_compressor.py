""" Module that computes the compressor for the fuel cell system in a hybrid propulsion model (FC-B configuration). """

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
from stdatm.atmosphere import Atmosphere
import numpy as np


class ComputeCompressor(om.ExplicitComponent):
    """
    Computes compressor power given cruising altitude, fuel cell design power, stack pressure and cell voltage.
    Formulas and assumptions are based on :
        https://www.researchgate.net/publication/319935703_A_Fuel_Cell_System_Sizing_Tool_Based_on_Current_Production_Aircraft
        https://repository.tudelft.nl/islandora/object/uuid%3A6e274095-9920-4d20-9e11-d5b76363e709
    Radius and weight computation is based on the work of S. Delbecq in "Compressor radius and mass estimation".
    Complemented with Cordier diagram "A theoretical derivation of the Cordier diagram for turbomachines", Ph Epple, F Durst, A. Delgado
    The sizing formulae should be revised.
    """

    def setup(self):
        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:stack_power", val=np.nan, units='W')
        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:stack_pressure", val=np.nan, units='Pa')
        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:cell_voltage", val=np.nan, units='V')
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units='m')
        self.add_input("data:propulsion:hybrid_powertrain:compressor:motor_efficiency", val=np.nan, units=None)
        self.add_input("data:propulsion:hybrid_powertrain:compressor:delta", val=np.nan, units=None)
        self.add_input("data:propulsion:hybrid_powertrain:compressor:specific_work", val=np.nan, units="J/kg")
        # self.add_input("data:propulsion:hybrid_powertrain:compressor:ref_mass", val=np.nan, units="kg")
        # self.add_input("data:propulsion:hybrid_powertrain:compressor:ref_radius", val=np.nan, units="m")

        self.add_output("data:propulsion:hybrid_powertrain:compressor:power", units='W')
        self.add_output("data:geometry:hybrid_powertrain:compressor:radius", units="m")
        # self.add_output("data:weight:hybrid_powertrain:compressor:mass", units="kg")

        self.declare_partials('*', '*', method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fc_design_power = inputs['data:propulsion:hybrid_powertrain:fuel_cell:stack_power']
        fc_cell_v = inputs['data:propulsion:hybrid_powertrain:fuel_cell:cell_voltage']
        fc_stack_pressure = inputs['data:propulsion:hybrid_powertrain:fuel_cell:stack_pressure']
        air_temp = Atmosphere(altitude=inputs['data:mission:sizing:main_route:cruise:altitude']).temperature  # [K]
        air_pressure = Atmosphere(altitude=inputs['data:mission:sizing:main_route:cruise:altitude']).pressure  # [Pa]
        air_density = Atmosphere(altitude=inputs['data:mission:sizing:main_route:cruise:altitude']).density  # [kg/m**3]
        motor_eff = inputs['data:propulsion:hybrid_powertrain:compressor:motor_efficiency']
        delta = inputs["data:propulsion:hybrid_powertrain:compressor:delta"]
        W = inputs["data:propulsion:hybrid_powertrain:compressor:specific_work"]
        # M_ref = inputs["data:propulsion:hybrid_powertrain:compressor:ref_mass"]
        # R_ref = inputs["data:propulsion:hybrid_powertrain:compressor:ref_radius"]

        # Determining air mass flow rate of the stack
        M_O2 = 31.998  # [g/mol] - Molar mass of oxygen
        stoich_ratio = 2  # Common oxygen stoichiometric ratio for the FC
        F = 96.485  # [C/mol] - Faraday Constant

        ox_flow = M_O2 / 1000 * fc_design_power * stoich_ratio / (4 * fc_cell_v * F) / 1000  # [kg/s]
        air_flow = ox_flow / 0.21  # [kg/s] - Assuming 21% of air is oxygen

        # Defining constants
        cp = 1004  # [J/kgK] - Specific heat capacity of air
        eta_c = 0.72  # Isentropic compressor efficiency taken at 72% by default
        gamma = 1.4  # Ratio of specific heat capacities of air
        if fc_stack_pressure > air_pressure:
            p_ratio = fc_stack_pressure / air_pressure
        else:
            raise ValueError("Fuel stack pressure is lower than ambient pressure,"
                             "increase the value of fuel_cell:stack_power")

        # Determining characteristics
        W = cp * air_temp * (p_ratio ** ((gamma - 1) / gamma) - 1) / eta_c
        specific_head = air_pressure*(p_ratio-1)/air_density
        # Determining compressor power
        compressor_power = motor_eff**(-1) * air_flow * W

        outputs['data:propulsion:hybrid_powertrain:compressor:power'] = compressor_power

        # Determining compressor radius and weight
        N_spe = 0.8
        Q_v = air_flow / air_density  # [m**3/s] - Volumetric flow
        k = 2 ** 0.25 * np.pi ** 0.5
        # sigma = N * (k / 60) * Q_v ** (0.5) / W ** (0.75)
        sigma = 0.15
        delta = 10 ** (0.23953) * sigma ** (
                -0.38527
                + 0.14978 * np.log10(sigma)
                - 0.16545 * np.log10(sigma) ** 2
                + 0.06385 * np.log10(sigma) ** 3
        )
        R = delta * Q_v ** 0.5 / (k * specific_head ** 0.25)
        # Not used, rotational speed formulae to verify
        N = sigma * (specific_head) ** (3 / 4) / (Q_v*1000) ** 0.5

        outputs["data:geometry:hybrid_powertrain:compressor:radius"] = R


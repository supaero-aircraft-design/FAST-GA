""" Module that sizes H2 storage in a hybrid propulsion model (FC-B configuration). """

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
import numpy as np
import math
from stdatm.atmosphere import Atmosphere


class ComputeH2Storage(om.ExplicitComponent):
    """
    For general aviation applications, 350 bar gaseous hydrogen storage methods are considered.
    Cylindrical tanks are computed.
    Code is based on the work done in 'FAST-GA-AMPERE' and on the storage model found here :
        https://www.researchgate.net/publication/24316784_Hydrogen_Storage_for_Aircraft_Applications_Overview
    """

    def setup(self):
        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:cell_voltage", val=np.nan, units='V')
        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:design_power", val=np.nan, units='W')
        self.add_input("data:mission:sizing:endurance", val=np.nan, units='min')
        self.add_input("data:propulsion:hybrid_powertrain:h2_storage:pressure", val=np.nan, units='Pa')
        self.add_input("data:geometry:hybrid_powertrain:h2_storage:nb_tanks", val=np.nan, units=None)
        self.add_input("data:geometry:hybrid_powertrain:h2_storage:length_radius_ratio", val=np.nan, units=None)
        self.add_input("data:geometry:hybrid_powertrain:h2_storage:fos", val=2.25, units=None,
                       desc='Factor of safety defined by industry standard specifications')
        self.add_input("data:geometry:hybrid_powertrain:h2_storage:maximum_stress", val=np.nan, units='Pa',
                       desc='Maximum stress allowed by the chosen material')
        self.add_input("data:geometry:hybrid_powertrain:h2_storage:thickness_fitting_factor", val=1, units=None,
                       desc='Parameter to adjust the thickness of the fuel tanks (too low)')
        self.add_input("data:geometry:hybrid_powertrain:h2_storage:tank_density", val=np.nan, units='kg/m**3')

        self.add_output("data:geometry:hybrid_powertrain:h2_storage:total_tanks_volume", units='m**3',
                        desc='Total volume of the tank(s)')
        self.add_output("data:geometry:hybrid_powertrain:h2_storage:single_tank_volume", units='m**3')
        self.add_output("data:geometry:hybrid_powertrain:h2_storage:tank_internal_volume", units='m**3')
        self.add_output("data:geometry:hybrid_powertrain:h2_storage:tank_internal_radius", units='m')
        self.add_output("data:geometry:hybrid_powertrain:h2_storage:tank_internal_length", units='m')
        self.add_output("data:geometry:hybrid_powertrain:h2_storage:wall_thickness", units='m')
        # self.add_output("data:weight:hybrid_powertrain:h2_storage:single_tank_mass", units='kg')
        # self.add_output("data:weight:hybrid_powertrain:h2_storage:total_tanks_mass", units='kg')

        self.declare_partials('*', '*', method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        V_cell = inputs['data:propulsion:hybrid_powertrain:fuel_cell:cell_voltage']
        required_power = inputs['data:propulsion:hybrid_powertrain:fuel_cell:design_power']
        op_time = inputs['data:mission:sizing:endurance'] * 60  # [s]
        P_H = inputs['data:propulsion:hybrid_powertrain:h2_storage:pressure']
        # T_H = inputs['data:propulsion:hybrid_powertrain:h2_storage:temperature']
        nb_tanks = inputs['data:geometry:hybrid_powertrain:h2_storage:nb_tanks']
        tank_lr_ratio = inputs['data:geometry:hybrid_powertrain:h2_storage:length_radius_ratio']  # length to radius
        FoS = inputs['data:geometry:hybrid_powertrain:h2_storage:fos']  # Factor of safety
        max_stress = inputs['data:geometry:hybrid_powertrain:h2_storage:maximum_stress']
        # density = inputs['data:geometry:hybrid_powertrain:h2_storage:tank_density']
        t_fit = inputs['data:geometry:hybrid_powertrain:h2_storage:thickness_fitting_factor']

        T_H = Atmosphere(altitude=0).temperature  # [K]

        # Determining total mass of hydrogen needed
        F_H = required_power / (V_cell * 2 * 96500 * 500)  # [kg/s] - Flow rate of hydrogen
        m_H = op_time * F_H  # [kg]

        # Determining volume of hydrogen needed
        Z = 0.99704 + 6.4149e-9 * P_H  # Hydrogen compressibility factor
        R = 4157.2  # [Nm/(Kkg)]
        V_H = Z * R * m_H * T_H / P_H  # [m**3]

        # Determining internal radius-length of a single cylindrical tank
        V_tank_int = V_H / nb_tanks
        tank_radius = (V_tank_int / (tank_lr_ratio * math.pi)) ** (1 / 3)  # [m]
        tank_length = tank_lr_ratio * tank_radius  # [m]

        # Determining wall thickness and tank volume
        thickness = P_H * tank_radius * FoS / (2 * max_stress) * t_fit  # [m]
        tank_ex_radius = tank_radius + thickness
        tank_ex_length = tank_length + 2 * thickness
        tank_volume = math.pi * (tank_ex_radius ** 2) * tank_ex_length  # [m**3]
        tot_tank_volume = tank_volume * nb_tanks  # [m**3]

        # Determining tank(s) mass : a fitting parameter is added to adjust the results
        # tank_mass = (tank_volume - V_tank_int) * density  # [kg]
        # tot_tank_mass = nb_tanks * tank_mass

        outputs['data:geometry:hybrid_powertrain:h2_storage:total_tanks_volume'] = tot_tank_volume
        outputs['data:geometry:hybrid_powertrain:h2_storage:single_tank_volume'] = tank_volume
        outputs['data:geometry:hybrid_powertrain:h2_storage:tank_internal_volume'] = V_tank_int
        outputs['data:geometry:hybrid_powertrain:h2_storage:tank_internal_radius'] = tank_radius
        outputs['data:geometry:hybrid_powertrain:h2_storage:tank_internal_length'] = tank_length
        outputs['data:geometry:hybrid_powertrain:h2_storage:wall_thickness'] = thickness
        # outputs['data:weight:hybrid_powertrain:h2_storage:single_tank_mass'] = tank_mass
        # outputs['data:weight:hybrid_powertrain:h2_storage:total_tanks_mass'] = tot_tank_mass

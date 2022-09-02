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
from .resources.h2_storage import GH2_storage

class ComputeH2Storage(om.ExplicitComponent):
    """
    For general aviation applications, 350 bar gaseous hydrogen storage methods are considered.
    Cylindrical tanks are computed.
    Code is based on the work done in 'FAST-GA-AMPERE' and on the storage model found here :
        https://www.researchgate.net/publication/24316784_Hydrogen_Storage_for_Aircraft_Applications_Overview

    Computing hydrogen storage weight based on volumetric index interval for two reference tanks of 350b and 700b:
    "Technical Assessment of Compressed Hydrogen Storage Tank Systems for Automotive Applications", Thanh Hua, Argonne National Lab
    """

    def initialize(self):
        self.options.declare('H2_storage_model',types=str, default='legacy')

    def setup(self):

        self.add_input("data:mission:sizing:fuel", val=np.nan, units='kg')
        self.add_input("data:propulsion:hybrid_powertrain:h2_storage:pressure", val=np.nan, units='MPa')
        self.add_input("data:geometry:hybrid_powertrain:h2_storage:nb_tanks", val=np.nan, units=None)
        self.add_input("data:geometry:hybrid_powertrain:h2_storage:length_radius_ratio", val=np.nan, units=None)

        model = self.options["H2_storage_model"]
        if model == "physical":
            self.add_input("data:geometry:hybrid_powertrain:h2_storage:fos", val=2.25, units=None,
                           desc='Factor of safety defined by industry standard specifications')
            self.add_input("data:geometry:hybrid_powertrain:h2_storage:maximum_stress", val=np.nan, units='Pa',
                           desc='Maximum stress allowed by the chosen material')
            self.add_input("data:geometry:hybrid_powertrain:h2_storage:thickness_fitting_factor", val=1, units=None,
                           desc='Parameter to adjust the thickness of the fuel tanks (too low)')
            self.add_input("data:geometry:hybrid_powertrain:h2_storage:liner_thickness", val=np.nan, units="m")

            self.add_output("data:geometry:hybrid_powertrain:h2_storage:tank_internal_radius", units='m')
            self.add_output("data:geometry:hybrid_powertrain:h2_storage:tank_internal_length", units='m')
            self.add_output("data:geometry:hybrid_powertrain:h2_storage:wall_thickness", units='m')
            self.add_output("data:geometry:hybrid_powertrain:h2_storage:single_tank_liner_volume", units='m**3')
        else:
            self.add_input("data:geometry:hybrid_powertrain:h2_storage:volumetric_capacity_350b", val=np.nan, units='kg/m**3')
            self.add_input("data:geometry:hybrid_powertrain:h2_storage:volumetric_capacity_700b", val=np.nan, units='kg/m**3')

        self.add_output("data:geometry:hybrid_powertrain:h2_storage:tank_internal_volume", units='m**3')
        self.add_output("data:geometry:hybrid_powertrain:h2_storage:total_tanks_volume", units='m**3',
                        desc='Total volume of the tank(s)')
        self.add_output("data:geometry:hybrid_powertrain:h2_storage:tank_ext_length", units='m')
        self.add_output("data:geometry:hybrid_powertrain:h2_storage:tank_ext_diameter", units='m')
        self.add_output("data:geometry:hybrid_powertrain:h2_storage:single_tank_volume", units='m**3')
        self.add_output("data:geometry:hybrid_powertrain:h2_storage:total_h2_mass_storable", units='kg')
        self.add_output("data:weight:aircraft:MFW", units="kg")

        self.declare_partials('*', '*', method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        model = self.options['H2_storage_model']

        if model == "physical":
            P_H = inputs['data:propulsion:hybrid_powertrain:h2_storage:pressure']
            nb_tanks = inputs['data:geometry:hybrid_powertrain:h2_storage:nb_tanks']
            tank_lr_ratio = inputs['data:geometry:hybrid_powertrain:h2_storage:length_radius_ratio']  # length to radius
            FoS = inputs['data:geometry:hybrid_powertrain:h2_storage:fos']  # Factor of safety
            max_stress = inputs['data:geometry:hybrid_powertrain:h2_storage:maximum_stress']
            t_fit = inputs['data:geometry:hybrid_powertrain:h2_storage:thickness_fitting_factor']
            m_fuel = inputs["data:mission:sizing:fuel"]
            liner_thick = inputs["data:geometry:hybrid_powertrain:h2_storage:liner_thickness"]

            storage_sys = GH2_storage(storage_pressure=P_H, n_tanks=nb_tanks, tank_lr_ratio=tank_lr_ratio,
                                      FoS=FoS, max_stress=max_stress, t_fit=t_fit, liner_thick=liner_thick)

            tank_length, tank_radius, tank_ext_volume = storage_sys.compute_geometry_physical_model(m_fuel)

            outputs['data:geometry:hybrid_powertrain:h2_storage:total_tanks_volume'] = tank_ext_volume * nb_tanks
            outputs['data:geometry:hybrid_powertrain:h2_storage:single_tank_volume'] = tank_ext_volume
            outputs['data:geometry:hybrid_powertrain:h2_storage:single_tank_liner_volume'] = storage_sys.liner_volume
            outputs['data:geometry:hybrid_powertrain:h2_storage:tank_internal_volume'] = storage_sys.tank_int_volume
            outputs['data:geometry:hybrid_powertrain:h2_storage:tank_internal_radius'] = storage_sys.tank_int_radius
            outputs['data:geometry:hybrid_powertrain:h2_storage:tank_internal_length'] = \
                storage_sys.tank_int_length + 2 * storage_sys.tank_int_radius
            outputs['data:geometry:hybrid_powertrain:h2_storage:tank_ext_diameter'] = tank_radius * 2
            outputs['data:geometry:hybrid_powertrain:h2_storage:tank_ext_diameter']= tank_length
            outputs['data:geometry:hybrid_powertrain:h2_storage:wall_thickness'] = storage_sys.thickness
            outputs['data:geometry:hybrid_powertrain:h2_storage:total_h2_mass_storable'] = m_fuel
            outputs['data:weight:aircraft:MFW'] = m_fuel

        else:
            P_H = inputs['data:propulsion:hybrid_powertrain:h2_storage:pressure']
            volu_index_35 = inputs["data:geometry:hybrid_powertrain:h2_storage:volumetric_capacity_350b"]
            volu_index_70 = inputs["data:geometry:hybrid_powertrain:h2_storage:volumetric_capacity_700b"]
            nb_tanks = inputs['data:geometry:hybrid_powertrain:h2_storage:nb_tanks']
            tank_lr_ratio = inputs['data:geometry:hybrid_powertrain:h2_storage:length_radius_ratio']  # length to radius
            m_fuel = inputs["data:mission:sizing:fuel"]

            storage_sys = GH2_storage(storage_pressure=P_H, n_tanks=nb_tanks, tank_lr_ratio=tank_lr_ratio, vol_cap_350=volu_index_35,
                                      vol_cap_700=volu_index_70)

            tank_length, tank_radius, tank_ext_volume = storage_sys.compute_geometry_volumetric_model(m_fuel)

            outputs['data:geometry:hybrid_powertrain:h2_storage:tank_ext_diameter'] = tank_radius * 2
            outputs['data:geometry:hybrid_powertrain:h2_storage:tank_ext_length'] = tank_length
            outputs['data:geometry:hybrid_powertrain:h2_storage:total_tanks_volume'] = tank_ext_volume * nb_tanks
            outputs['data:geometry:hybrid_powertrain:h2_storage:single_tank_volume'] = tank_ext_volume
            outputs['data:geometry:hybrid_powertrain:h2_storage:tank_internal_volume'] = storage_sys.tank_int_volume
            outputs['data:weight:aircraft:MFW'] = m_fuel
            outputs['data:geometry:hybrid_powertrain:h2_storage:total_h2_mass_storable'] = m_fuel

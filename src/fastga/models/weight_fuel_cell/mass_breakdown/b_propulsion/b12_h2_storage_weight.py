""" Estimation of hydrogen tanks weight """
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
from openmdao.core.explicitcomponent import ExplicitComponent
from fastoad.module_management.service_registry import RegisterSubmodel
from .constants import SUBMODEL_PROPULSION_H2STORAGE_MASS

@RegisterSubmodel(SUBMODEL_PROPULSION_H2STORAGE_MASS, "fastga.submodel.weight.mass.propulsion.hybrid.fuelcell.h2storage.legacy")
class ComputeH2StorageWeightLegacy(ExplicitComponent):
    """
    Computing hydrogen storage weight based on gravimetric index interval for two reference tanks of 350b and 700b:
    "Technical Assessment of Compressed Hydrogen Storage Tank Systems for Automotive Applications", Thanh Hua, Argonne National Lab
    """
    def setup(self):

        self.add_input("data:geometry:hybrid_powertrain:h2_storage:gravimetric_capacity_350b", val=np.nan)
        self.add_input("data:geometry:hybrid_powertrain:h2_storage:gravimetric_capacity_700b",val=np.nan)
        self.add_input("data:propulsion:hybrid_powertrain:h2_storage:pressure",val = np.nan, units='MPa')
        self.add_input("data:mission:sizing:fuel", val=np.nan, units='kg')

        self.add_output("data:weight:hybrid_powertrain:h2_storage:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        grav_cap_350 = inputs["data:geometry:hybrid_powertrain:h2_storage:gravimetric_capacity_350b"]
        grav_cap_700 = inputs["data:geometry:hybrid_powertrain:h2_storage:gravimetric_capacity_700b"]
        pressure = inputs["data:propulsion:hybrid_powertrain:h2_storage:pressure"]
        fuel_weight = inputs["data:mission:sizing:fuel"]

        # Interpolation
        if pressure < 35:
            grav_cap = grav_cap_350
        elif pressure > 70:
            grav_cap = grav_cap_700
        else:
            grav_cap = np.interp(pressure,[35,70],np.concatenate((grav_cap_350,grav_cap_700)))

        b12 = fuel_weight / grav_cap


        outputs['data:weight:hybrid_powertrain:h2_storage:mass'] = b12


@RegisterSubmodel(SUBMODEL_PROPULSION_H2STORAGE_MASS, "fastga.submodel.weight.mass.propulsion.hybrid.fuelcell.h2storage.physical")
class ComputeH2StorageWeightPhysical(ExplicitComponent):
        """
        Computing hydrogen storage weight using physical model of A. J. COLOZZA. “Hydrogen Storage for Aircraft Applications Overview”. In: (2002).
        Complemented and calibrated based on data of "Technical Assessment of Compressed Hydrogen Storage Tank Systems for Automotive Applications", Thanh Hua, Argonne National Lab
        """

        def setup(self):

            self.add_input("data:geometry:hybrid_powertrain:h2_storage:nb_tanks", val=np.nan, units=None)
            self.add_input("data:geometry:hybrid_powertrain:h2_storage:single_tank_volume", val=np.nan, units='m**3')
            self.add_input("data:geometry:hybrid_powertrain:h2_storage:tank_internal_volume", val=np.nan, units="m**3")
            self.add_input("data:geometry:hybrid_powertrain:h2_storage:cfc_density", val=np.nan, units='kg/m**3')
            self.add_input("data:geometry:hybrid_powertrain:h2_storage:liner_density", val=np.nan, units='kg/m**3')
            self.add_input("data:geometry:hybrid_powertrain:h2_storage:bop_factor", val=np.nan, units=None)
            self.add_input("data:geometry:hybrid_powertrain:h2_storage:single_tank_liner_volume", val=np.nan, units='m**3')

            self.add_output("data:weight:hybrid_powertrain:h2_storage:mass", units="kg")

        def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

            nb_tanks = inputs['data:geometry:hybrid_powertrain:h2_storage:nb_tanks']
            tank_volume = inputs['data:geometry:hybrid_powertrain:h2_storage:single_tank_volume']
            density_cfp = inputs['data:geometry:hybrid_powertrain:h2_storage:cfc_density']
            density_liner = inputs['data:geometry:hybrid_powertrain:h2_storage:liner_density']
            bop_factor = inputs["data:geometry:hybrid_powertrain:h2_storage:bop_factor"]
            liner_volum = inputs["data:geometry:hybrid_powertrain:h2_storage:single_tank_liner_volume"]
            int_volume = inputs['data:geometry:hybrid_powertrain:h2_storage:tank_internal_volume']

            tank_mass = ((tank_volume - liner_volum - int_volume) * density_cfp + liner_volum * density_liner) * bop_factor  # [kg]

            b12 = nb_tanks * tank_mass

            outputs['data:weight:hybrid_powertrain:h2_storage:mass'] = b12

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


class ComputeH2StorageWeight(ExplicitComponent):
    """
    Computing hydrogen storage weight
    """
    def setup(self):

        self.add_input("data:geometry:hybrid_powertrain:h2_storage:nb_tanks", val=np.nan, units=None)
        self.add_input("data:geometry:hybrid_powertrain:h2_storage:single_tank_volume", val=np.nan, units='m**3')
        self.add_input("data:geometry:hybrid_powertrain:h2_storage:tank_internal_volume", val=np.nan, units="m**3")
        self.add_input("data:geometry:hybrid_powertrain:h2_storage:tank_density", val=np.nan, units='kg/m**3')
        # self.add_input("data:geometry:hybrid_powertrain:h2_storage:mass_fitting_factor", val=1, units=None,
        #                desc='Parameter to adjust the mass of the fuel tanks arguably too high')

        self.add_output("data:weight:hybrid_powertrain:h2_storage:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        nb_tanks = inputs['data:geometry:hybrid_powertrain:h2_storage:nb_tanks']
        tank_volume = inputs['data:geometry:hybrid_powertrain:h2_storage:single_tank_volume']
        int_volume = inputs['data:geometry:hybrid_powertrain:h2_storage:tank_internal_volume']
        density = inputs['data:geometry:hybrid_powertrain:h2_storage:tank_density']
        # mass_fit = inputs['data:geometry:hybrid_powertrain:h2_storage:mass_fitting_factor']

        b12 = nb_tanks * (tank_volume - int_volume) * density  # [kg]

        outputs['data:weight:hybrid_powertrain:h2_storage:mass'] = b12

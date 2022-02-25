""" Module that computes the heat exchanger subsystem in a hybrid propulsion model (FC-B configuration). """

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
from stdatm.atmosphere import Atmosphere


class ComputeHex(om.ExplicitComponent):
    """
    This discipline computes the Heat Exchanger assuming it to be a Compact Heat Exchanger (CHE).
    Based on :
        - work done in FAST-GA-AMPERE
        - https://apps.dtic.mil/sti/pdfs/ADA525161.pdf
    The HEX subsystem cools the waste water-air mixture produced by the fuel cell stacks.
    """

    def setup(self):

        self.add_input("data:propulsion:hybrid_powertrain:hex:air_speed", val=np.nan, units='m/s')
        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:operating_temperature", val=np.nan, units='K')
        # self.add_input("data:geometry:hybrid_powertrain:hex:radiator_surface_density", val=np.nan, units='g/cm**2')
        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:cooling_power", val=np.nan, units='W')
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units='m')

        self.add_output("data:geometry:hybrid_powertrain:hex:area", units="m**2")
        # self.add_output("data:weight:hybrid_powertrain:hex:radiator_mass", units='kg')

        self.declare_partials('*', '*', method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        air_speed = inputs['data:propulsion:hybrid_powertrain:hex:air_speed']
        op_T = inputs['data:propulsion:hybrid_powertrain:fuel_cell:operating_temperature']
        # surface_density = inputs['data:geometry:hybrid_powertrain:hex:radiator_surface_density']
        fc_cooling_power = inputs['data:propulsion:hybrid_powertrain:fuel_cell:cooling_power']
        ext_T = Atmosphere(altitude=inputs['data:mission:sizing:main_route:cruise:altitude']).temperature  # [K]

        # Determining temperature gap and dissipative power of the CHE
        delta_T = op_T - ext_T
        h = 1269.0 * air_speed + 99.9  # [W/(m**2K)] - Heat Transfer Coefficient used in FAST-GA-AMPERE - Based on a
        # correlation described in 'Design upgrade and Performance assessment of the AMPERE Distributed Electric
        # Propulsion concept' - F. Lutz

        # Determining surface of the radiator
        needed_area = fc_cooling_power / (h * delta_T)  # [m**2]
        outputs['data:geometry:hybrid_powertrain:hex:area'] = needed_area

        # Determining mass of the radiator
        # M_rad = needed_area * 10000 * surface_density
        # outputs['data:weight:hybrid_powertrain:hex:radiator_mass'] = M_rad

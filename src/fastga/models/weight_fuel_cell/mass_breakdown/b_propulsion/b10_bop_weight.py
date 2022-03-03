""" Estimation of Balance of Plant weight for the fuel cells system. It also includes Heat Exchanger and compressor
weight calculations. """
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


class ComputeBoPWeight(ExplicitComponent):
    """
    Weight estimation for the BoP.
    """

    def setup(self):
        self.add_input("data:propulsion:hybrid_powertrain:compressor:ref_mass", val=np.nan, units="kg")
        self.add_input("data:propulsion:hybrid_powertrain:compressor:ref_radius", val=np.nan, units="m")
        self.add_input("data:geometry:hybrid_powertrain:compressor:radius", val=np.nan, units="m")
        self.add_input("data:geometry:hybrid_powertrain:hex:radiator_surface_density", val=np.nan, units='kg/cm**2')
        self.add_input("data:geometry:hybrid_powertrain:hex:area", val=np.nan, units='m**2')
        self.add_input("data:weight:hybrid_powertrain:fuel_cell:mass", val=np.nan, units="kg")

        self.add_output("data:weight:hybrid_powertrain:compressor:mass", units="kg")
        self.add_output("data:weight:hybrid_powertrain:hex:radiator_mass", units="kg")
        self.add_output("data:weight:hybrid_powertrain:bop:lc_ss_mass", units="kg")
        self.add_output("data:weight:hybrid_powertrain:bop:total_mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        R = inputs['data:geometry:hybrid_powertrain:compressor:radius']
        M_ref = inputs["data:propulsion:hybrid_powertrain:compressor:ref_mass"]
        R_ref = inputs["data:propulsion:hybrid_powertrain:compressor:ref_radius"]

        radiator_area = inputs['data:geometry:hybrid_powertrain:hex:area']
        radiator_surface_density = inputs['data:geometry:hybrid_powertrain:hex:radiator_surface_density']

        stack_mass = inputs['data:weight:hybrid_powertrain:fuel_cell:mass']

        # Compressor
        M_compressor = M_ref * (R / R_ref) ** 3

        outputs['data:weight:hybrid_powertrain:compressor:mass'] = M_compressor

        # HEX
        M_hex = radiator_area * 10000 * radiator_surface_density

        outputs['data:weight:hybrid_powertrain:hex:radiator_mass'] = M_hex

        # FC liquid cooling subsystem : based on https://www.researchgate.net/publication/319935703
        LC_SUBSYSTEM_MASS_FRACTION = 0.17
        M_lcss = LC_SUBSYSTEM_MASS_FRACTION * stack_mass

        outputs['data:weight:hybrid_powertrain:bop:lc_ss_mass'] = M_lcss

        # Total mass of the BoP
        b10 = M_compressor + M_hex + M_lcss

        outputs['data:weight:hybrid_powertrain:bop:total_mass'] = b10

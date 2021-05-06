"""
    Estimation of aerodynamic center
"""

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
import math
from openmdao.core.explicitcomponent import ExplicitComponent

from fastoad.model_base import Atmosphere


class ComputeAeroCenter(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Aerodynamic center estimation """

    def setup(self):
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:elevator:low_speed:CL_delta", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:horizontal_tail:efficiency", val=np.nan)
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")

        self.add_output("data:aerodynamics:cruise:neutral_point:stick_fixed:x")
        self.add_output("data:aerodynamics:cruise:neutral_point:stick_free:x")
        self.add_output("data:aerodynamics:cruise:neutral_point:free_elevator_factor")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        x0_wing = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        fus_length = inputs["data:geometry:fuselage:length"]
        wing_area = inputs["data:geometry:wing:area"]
        aspect_ratio = inputs["data:geometry:wing:aspect_ratio"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        cl_alpha_ht = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
        cl_delta_ht = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]
        ch_alpha_3d = inputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha"]
        ch_delta_3d = inputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta"]
        tail_efficiency = inputs["data:aerodynamics:horizontal_tail:efficiency"]
        v_cruise = inputs["data:TLAR:v_cruise"]
        alt_cruise = inputs["data:mission:sizing:main_route:cruise:altitude"]

        # TODO: make variable name in computation sequence more english
        x0_25 = fa_length - 0.25 * l0_wing - x0_wing + 0.25 * l1_wing
        ratio_x025 = x0_25 / fus_length
        # fitting result of Raymer book, figure 16.14
        k_h = 0.01222 - 7.40541e-4 * ratio_x025 * 100 + 2.1956e-5 * (ratio_x025 * 100) ** 2
        # equation from Raymer book, eqn 16.22
        # FIXME: introduce cm_alpha_wing to the equation (non-symmetrical profile)
        cm_alpha_fus = k_h * width_max ** 2 * fus_length / (l0_wing * wing_area) * 180.0 / np.pi
        x_ca_plane = (tail_efficiency * cl_alpha_ht * lp_ht - cm_alpha_fus * l0_wing) / \
                     (cl_alpha_wing + tail_efficiency * cl_alpha_ht)
        x_aero_center = x_ca_plane / l0_wing + 0.25

        outputs["data:aerodynamics:cruise:neutral_point:stick_fixed:x"] = x_aero_center

        sos = Atmosphere(alt_cruise).speed_of_sound
        mach = v_cruise / sos
        beta = math.sqrt(1. - mach ** 2.0)
        cl_delta_ht_cruise = cl_delta_ht / beta

        # The cl_alpha_ht in the formula for the free_elevator_factor is defined with respect to the tail angle of
        # attack, the one we compute is wth respect to the plane so it includes downwash, as a consequence we must
        # correct it influence for this specific calculation. We will use the formula for elliptical wing as it is well
        # known
        downwash_effect = (1. - 2. * cl_alpha_wing / (math.pi * aspect_ratio))
        cl_alpha_ht_ht = cl_alpha_ht / downwash_effect
        free_elevator_factor = 1. - (cl_delta_ht_cruise/cl_alpha_ht_ht)*(ch_alpha_3d/ch_delta_3d)

        outputs["data:aerodynamics:cruise:neutral_point:free_elevator_factor"] = free_elevator_factor

        x_ca_plane_free = (tail_efficiency * free_elevator_factor * cl_alpha_ht * lp_ht - cm_alpha_fus * l0_wing) / \
                          (cl_alpha_wing + tail_efficiency * free_elevator_factor * cl_alpha_ht)
        x_aero_center_free = x_ca_plane_free / l0_wing + 0.25

        outputs["data:aerodynamics:cruise:neutral_point:stick_free:x"] = x_aero_center_free

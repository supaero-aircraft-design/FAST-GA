#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
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
import openmdao.api as om
import fastoad.api as oad

from .compute_cl_wing import ComputeWingLiftCoefficient
from ..digitization.compute_cn_r_wing_lift_effect import ComputeWingLiftEffectCnr
from ..digitization.compute_cn_r_wing_drag_effect import ComputeWingDragEffectCnr

from ...constants import SUBMODEL_CN_R_WING


@oad.RegisterSubmodel(
    SUBMODEL_CN_R_WING, "fastga.submodel.aerodynamics.wing.yaw_moment_yaw_rate.legacy"
)
class ComputeCnYawRateWing(om.Group):
    """
    Class to compute the contribution of the wing to the yaw moment coefficient due to yaw rate (
    yaw damping). Depends on the lift coefficient of the wing, hence on the reference angle of
    attack, so the same remark as in ..compute_cy_yaw_rate.py holds. The convention from
    :cite:`roskampart6:1985` are used, meaning that for lateral derivative, the reference length
    is the wing span. Another important point is that, for the derivative with respect to yaw and
    roll, the rotation speed are made dimensionless by multiplying them by the wing span and
    dividing them by 2 times the airspeed.

    Based on :cite:`roskampart6:1985` section 10.2.8
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_subsystem(
            name="wing_lift_coeff_" + ls_tag,
            subsys=ComputeWingLiftCoefficient(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["data:*", "settings:*"],
        )
        self.add_subsystem(
            name="lift_effect_" + ls_tag,
            subsys=ComputeWingLiftEffectCnr(),
            promotes=["data:*"],
        )
        self.add_subsystem(
            name="drag_effect_" + ls_tag,
            subsys=ComputeWingDragEffectCnr(),
            promotes=["data:*"],
        )
        self.add_subsystem(
            name="cn_yaw_wing_" + ls_tag,
            subsys=_ComputeCnYawRateWing(),
            promotes=["data:*"],
        )

        self.connect("wing_lift_coeff_" + ls_tag + ".CL_wing", "cn_yaw_wing_" + ls_tag + ".CL_wing")
        self.connect(
            "lift_effect_" + ls_tag + ".lift_effect", "cn_yaw_wing_" + ls_tag + ".lift_effect"
        )
        self.connect(
            "drag_effect_" + ls_tag + ".drag_effect", "cn_yaw_wing_" + ls_tag + ".drag_effect"
        )


class _ComputeCnYawRateWing(om.ExplicitComponent):
    """
    Class to compute the contribution of the wing to the yaw moment coefficient due to yaw rate (
    yaw damping). Depends on the lift coefficient of the wing, hence on the reference angle of
    attack, so the same remark as in ..compute_cy_yaw_rate.py holds. The convention from
    :cite:`roskampart6:1985` are used, meaning that for lateral derivative, the reference length
    is the wing span. Another important point is that, for the derivative with respect to yaw and
    roll, the rotation speed are made dimensionless by multiplying them by the wing span and
    dividing them by 2 times the airspeed.

    Based on :cite:`roskampart6:1985` section 10.2.8
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO initialize
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input("drag_effect", val=np.nan, units="unitless")
        self.add_input("lift_effect", val=np.nan, units="unitless")
        self.add_input("CL_wing", val=np.nan, units="unitless")
        self.add_input("data:aerodynamics:wing:" + ls_tag + ":CD0", val=np.nan)

        self.add_output("data:aerodynamics:wing:" + ls_tag + ":Cn_r", units="rad**-1")

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.declare_partials(
            of="data:aerodynamics:wing:" + ls_tag + ":Cn_r", wrt="*", method="exact"
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        # Fuselage contribution neglected
        cl_w = inputs["CL_wing"]
        cd_0_wing = inputs["data:aerodynamics:wing:" + ls_tag + ":CD0"]
        lift_effect = inputs["lift_effect"]
        drag_effect = inputs["drag_effect"]

        outputs["data:aerodynamics:wing:" + ls_tag + ":Cn_r"] = (
            lift_effect * cl_w**2.0 + drag_effect * cd_0_wing
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        cl_w = inputs["CL_wing"]
        cd_0_wing = inputs["data:aerodynamics:wing:" + ls_tag + ":CD0"]
        lift_effect = inputs["lift_effect"]
        drag_effect = inputs["drag_effect"]

        partials["data:aerodynamics:wing:" + ls_tag + ":Cn_r", "lift_effect"] = cl_w**2.0

        partials["data:aerodynamics:wing:" + ls_tag + ":Cn_r", "CL_wing"] = 2.0 * lift_effect * cl_w

        partials["data:aerodynamics:wing:" + ls_tag + ":Cn_r", "drag_effect"] = cd_0_wing

        partials[
            "data:aerodynamics:wing:" + ls_tag + ":Cn_r",
            "data:aerodynamics:wing:" + ls_tag + ":CD0",
        ] = drag_effect

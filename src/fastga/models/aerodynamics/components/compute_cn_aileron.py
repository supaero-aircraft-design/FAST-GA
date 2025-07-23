"""Estimation of rolling moment du to the ailerons."""
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
import fastoad.api as oad
import openmdao.api as om

from .wing.compute_cl_wing import ComputeWingLiftCoefficient
from .digitization.compute_cn_delta_a_correlation_constatnt import (
    ComputeAileronYawCorrelationConstant,
)
from ..constants import SUBMODEL_CN_AILERON


@oad.RegisterSubmodel(SUBMODEL_CN_AILERON, "fastga.submodel.aerodynamics.aileron.yaw_moment.legacy")
class ComputeCnDeltaAileron(om.Group):
    """
    Yaw moment due to aileron deflection (also called adverse aileron yaw). Depends on the wing
    lift, hence on the angle of attack, so the same remark as in ..compute_cy_yaw_rate.py holds.
    The convention from :cite:`roskampart6:1985` are used, meaning that for lateral derivative,
    the reference length is the wing span.

    Based on :cite:`roskampart6:1985` section 10.3.5.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_subsystem(
            name="cl_w_" + ls_tag,
            subsys=ComputeWingLiftCoefficient(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["data:*", "settings:*"],
        )
        self.add_subsystem(
            name="correlation_constant_" + ls_tag,
            subsys=ComputeAileronYawCorrelationConstant(),
            promotes=["data:*"],
        )
        self.add_subsystem(
            name="cn_delta_aileron_" + ls_tag,
            subsys=ComputeCnDeltaAileronDeflection(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["data:*"],
        )

        self.connect("cl_w_" + ls_tag + ".CL_wing", "cn_delta_aileron_" + ls_tag + ".CL_wing")
        self.connect(
            "correlation_constant_" + ls_tag + ".aileron_correlation_constant",
            "cn_delta_aileron_" + ls_tag + ".aileron_correlation_constant",
        )


class ComputeCnDeltaAileronDeflection(om.ExplicitComponent):
    """
    Yaw moment due to aileron deflection (also called adverse aileron yaw). Depends on the wing
    lift, hence on the angle of attack, so the same remark as in ..compute_cy_yaw_rate.py holds.
    The convention from :cite:`roskampart6:1985` are used, meaning that for lateral derivative,
    the reference length is the wing span.

    Based on :cite:`roskampart6:1985` section 10.3.5.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO initialize
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input("CL_wing", val=np.nan, units="unitless")
        self.add_input("aileron_correlation_constant", val=np.nan, units="unitless")
        self.add_input(
            "data:aerodynamics:aileron:" + ls_tag + ":Cl_delta_a", val=np.nan, units="rad**-1"
        )

        self.add_output("data:aerodynamics:aileron:" + ls_tag + ":Cn_delta_a", units="rad**-1")

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        cl_delta_a = inputs["data:aerodynamics:aileron:" + ls_tag + ":Cl_delta_a"]
        cl_w = inputs["CL_wing"]
        correlation_constant = inputs["aileron_correlation_constant"]

        outputs["data:aerodynamics:aileron:" + ls_tag + ":Cn_delta_a"] = (
            correlation_constant * cl_w * cl_delta_a
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        cl_delta_a = inputs["data:aerodynamics:aileron:" + ls_tag + ":Cl_delta_a"]
        cl_w = inputs["CL_wing"]
        correlation_constant = inputs["aileron_correlation_constant"]

        partials[
            "data:aerodynamics:aileron:" + ls_tag + ":Cn_delta_a", "aileron_correlation_constant"
        ] = cl_w * cl_delta_a

        partials["data:aerodynamics:aileron:" + ls_tag + ":Cn_delta_a", "CL_wing"] = (
            correlation_constant * cl_delta_a
        )

        partials[
            "data:aerodynamics:aileron:" + ls_tag + ":Cn_delta_a",
            "data:aerodynamics:aileron:" + ls_tag + ":Cl_delta_a",
        ] = correlation_constant * cl_w

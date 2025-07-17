"""
Computation of wing lift coefficient in reference conditions, used only for the computation of
certain aerodynamic derivatives.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
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


class ComputeWingLiftCoefficient(om.ExplicitComponent):
    """
    Computation of the wing lift coefficient for use in the computation of aerodynamic derivatives.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO initialize
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"
        aoa = np.deg2rad(5.0) if self.options["low_speed_aero"] else np.deg2rad(1.0)

        self.add_input("data:aerodynamics:wing:" + ls_tag + ":CL0_clean", val=np.nan)
        self.add_input(
            "data:aerodynamics:wing:" + ls_tag + ":CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input(
            "settings:aerodynamics:reference_flight_conditions:" + ls_tag + ":AOA",
            units="rad",
            val=aoa,
        )

        self.add_output(
            "CL_wing",
            val=0.7,
            units="unitless",
            desc="Wing lift coefficient only applies in aerodynamic calculations",
        )

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.declare_partials(of="*", wrt="*", method="exact")
        self.declare_partials(
            of="CL_wing",
            wrt="data:aerodynamics:wing:" + ls_tag + ":CL0_clean",
            val=1.0,
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        outputs["CL_wing"] = (
            inputs["data:aerodynamics:wing:" + ls_tag + ":CL0_clean"]
            + inputs["data:aerodynamics:wing:" + ls_tag + ":CL_alpha"]
            * inputs["settings:aerodynamics:reference_flight_conditions:" + ls_tag + ":AOA"]
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        partials[
            "CL_wing",
            "data:aerodynamics:wing:" + ls_tag + ":CL_alpha",
        ] = inputs["settings:aerodynamics:reference_flight_conditions:" + ls_tag + ":AOA"]

        partials[
            "CL_wing",
            "settings:aerodynamics:reference_flight_conditions:" + ls_tag + ":AOA",
        ] = inputs["data:aerodynamics:wing:" + ls_tag + ":CL_alpha"]

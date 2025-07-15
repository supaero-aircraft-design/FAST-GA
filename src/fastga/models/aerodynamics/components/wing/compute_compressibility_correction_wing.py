"""Compressibility correction for swept wing."""
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


class ComputeCompressibilityCorrectionWing(om.ExplicitComponent):
    """
    Computation of compressibility correction factor for swept wing.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO initialize
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:aerodynamics:" + ls_tag + ":mach", val=np.nan)

        self.add_output("mach_correction_wing", val=1.0)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]  # In rad !!!
        mach = inputs["data:aerodynamics:" + ls_tag + ":mach"]

        outputs["mach_correction_wing"] = np.sqrt(1.0 - mach**2.0 * np.cos(wing_sweep_25) ** 2.0)

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]  # In rad !!!
        mach = inputs["data:aerodynamics:" + ls_tag + ":mach"]

        partials[
            "mach_correction_wing",
            "data:aerodynamics:" + ls_tag + ":mach",
        ] = (
            -mach
            * np.cos(wing_sweep_25) ** 2.0
            / np.sqrt(1.0 - mach**2.0 * np.cos(wing_sweep_25) ** 2.0)
        )

        partials["mach_correction_wing", "data:geometry:wing:sweep_25"] = (
            mach**2.0
            * np.cos(wing_sweep_25)
            * np.sin(wing_sweep_25)
            / np.sqrt(1.0 - mach**2.0 * np.cos(wing_sweep_25) ** 2.0)
        )

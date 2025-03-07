"""
Python module for navigation systems weight calculation, part of the systems mass computation.
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

import fastoad.api as oad
import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent

from .constants import (
    SERVICE_AVIONICS_SYSTEM_MASS,
    SUBMODEL_AVIONICS_SYSTEM_MASS_LEGACY,
    SUBMODEL_AVIONICS_SYSTEM_MASS_FROM_UNINSTALLED,
)

oad.RegisterSubmodel.active_models[SERVICE_AVIONICS_SYSTEM_MASS] = (
    SUBMODEL_AVIONICS_SYSTEM_MASS_LEGACY
)


@oad.RegisterSubmodel(
    SERVICE_AVIONICS_SYSTEM_MASS,
    SUBMODEL_AVIONICS_SYSTEM_MASS_LEGACY,
)
class ComputeAvionicsSystemsWeight(ExplicitComponent):
    """
    Weight estimation for avionics systems. Takes into account the weight of:
    - Instrumentation
    - Avionics
    - Electronics

    Based on a statistical analysis. See :cite:`roskampart5:1985` Torenbeek method. This method
    might not be suited for modern aircraft with EFIS type cockpit installation according to Roskam.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lbm")
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)
        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)

        self.add_output("data:weight:systems:avionics:mass", units="lbm")

        self.declare_partials(
            of="data:weight:systems:avionics:mass",
            wrt=["data:weight:aircraft:MTOW", "data:geometry:cabin:seats:passenger:NPAX_max"],
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mtow = inputs["data:weight:aircraft:MTOW"]
        n_eng = inputs["data:geometry:propulsion:engine:count"]
        n_pax = inputs["data:geometry:cabin:seats:passenger:NPAX_max"]

        n_occ = n_pax + 2.0
        # The formula differs depending on the number of propeller on the engine

        if n_eng == 1.0:
            c3 = 33.0 * n_occ

        else:
            c3 = 40 + 0.008 * mtow  # mass formula in lb

        outputs["data:weight:systems:avionics:mass"] = c3

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        n_eng = inputs["data:geometry:propulsion:engine:count"]

        if n_eng == 1.0:
            partials["data:weight:systems:avionics:mass", "data:weight:aircraft:MTOW"] = 0.0
            partials[
                "data:weight:systems:avionics:mass", "data:geometry:cabin:seats:passenger:NPAX_max"
            ] = 33.0
        else:
            partials["data:weight:systems:avionics:mass", "data:weight:aircraft:MTOW"] = 0.008
            partials[
                "data:weight:systems:avionics:mass", "data:geometry:cabin:seats:passenger:NPAX_max"
            ] = 0.0


@oad.RegisterSubmodel(
    SERVICE_AVIONICS_SYSTEM_MASS,
    SUBMODEL_AVIONICS_SYSTEM_MASS_FROM_UNINSTALLED,
)
class ComputeAvionicsSystemsWeightFromUninstalled(ExplicitComponent):
    """
    Weight estimation for avionics systems. Takes into account the weight of:
    - Instrumentation
    - Avionics
    - Navigation

    Based on a statistical analysis. See :cite:`gudmundsson:2013`.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input(
            "data:weight:systems:avionics:mass_uninstalled",
            val=45.0,
            units="lbm",
            desc="Weight of the uninstalled avionics system, default correspond to the value of "
            "the Garmin G1000",
        )

        self.add_output("data:weight:systems:avionics:mass", units="lbm")

        self.declare_partials("*", "*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        uninstalled_avionics = inputs["data:weight:systems:avionics:mass_uninstalled"]

        outputs["data:weight:systems:avionics:mass"] = 2.11 * uninstalled_avionics**0.933

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        uninstalled_avionics = inputs["data:weight:systems:avionics:mass_uninstalled"]

        partials[
            "data:weight:systems:avionics:mass", "data:weight:systems:avionics:mass_uninstalled"
        ] = 2.11 * 0.933 * uninstalled_avionics**-0.067

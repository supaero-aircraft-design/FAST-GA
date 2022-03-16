"""Estimation of the profile drag of miscellaneous items."""
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

from ..constants import SUBMODEL_CD0_OTHER


@RegisterSubmodel(SUBMODEL_CD0_OTHER, "fastga.submodel.aerodynamics.other.cd0.fuel_cell")
class Cd0Other(ExplicitComponent):
    """
    Profile drag estimation for miscellaneous items such as cowling, cooling and various component

    Based on : Gudmundsson, Snorri. General aviation aircraft design: Applied Methods and
    Procedures. Butterworth-Heinemann, 2013.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:aerodynamics:intakes:CD0", val = np.nan)

        if self.options["low_speed_aero"]:
            self.add_output("data:aerodynamics:other:low_speed:CD0")
        else:
            self.add_output("data:aerodynamics:other:cruise:CD0")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prop_layout = inputs["data:geometry:propulsion:engine:layout"]
        wing_area = inputs["data:geometry:wing:area"]

        # COWLING (only if engine in fuselage): cx_cowl*wing_area assumed typical (Gudmundsson p739)
        if prop_layout == 3.0:
            cd0_cowling = 0.0267 / wing_area
        else:
            cd0_cowling = 0.0

        # Drag due to intakes for cooling of FC calculated in hybrid_powertain
        cd0_cooling = (
            inputs["data:aerodynamics:intakes:CD0"]
        )

        # Gudmundsson p739. Sum of other components (not calculated here), cx_other*wing_area
        # assumed typical
        cd0_components = 0.0253 / wing_area

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:other:low_speed:CD0"] = (
                cd0_cowling + cd0_cooling + cd0_components
            )
        else:
            outputs["data:aerodynamics:other:cruise:CD0"] = (
                cd0_cowling + cd0_cooling + cd0_components
            )

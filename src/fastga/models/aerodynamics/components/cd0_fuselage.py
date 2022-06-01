"""Estimation of the fuselage profile drag."""
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

import math

import numpy as np
import fastoad.api as oad
from openmdao.core.explicitcomponent import ExplicitComponent

from ..constants import SUBMODEL_CD0_FUSELAGE


@oad.RegisterSubmodel(SUBMODEL_CD0_FUSELAGE, "fastga.submodel.aerodynamics.fuselage.cd0.legacy")
class Cd0Fuselage(ExplicitComponent):
    """
    Profile drag estimation for the fuselage

    Based on : Gudmundsson, Snorri. General aviation aircraft design: Applied Methods and
    Procedures. Butterworth-Heinemann, 2013.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:wet_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:unit_reynolds", val=np.nan, units="m**-1")
            self.add_output("data:aerodynamics:fuselage:low_speed:CD0")
        else:
            self.add_input("data:aerodynamics:cruise:unit_reynolds", val=np.nan, units="m**-1")
            self.add_output("data:aerodynamics:fuselage:cruise:CD0")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        height = inputs["data:geometry:fuselage:maximum_height"]
        width = inputs["data:geometry:fuselage:maximum_width"]
        length = inputs["data:geometry:fuselage:length"]
        wet_area_fus = inputs["data:geometry:fuselage:wet_area"]
        wing_area = inputs["data:geometry:wing:area"]
        if self.options["low_speed_aero"]:
            unit_reynolds = inputs["data:aerodynamics:low_speed:unit_reynolds"]
        else:
            unit_reynolds = inputs["data:aerodynamics:cruise:unit_reynolds"]

        # Local Reynolds:
        reynolds = unit_reynolds * length
        # 5% NLF
        x_trans = 0.05
        # Roots
        x0_turbulent = 36.9 * x_trans ** 0.625 * (1.0 / reynolds) ** 0.375
        cf_fus = 0.074 / reynolds ** 0.2 * (1.0 - (x_trans - x0_turbulent)) ** 0.8
        f = length / math.sqrt(4 * height * width / math.pi)
        ff_fus = 1.0 + 60.0 / (f ** 3.0) + f / 400.0
        # Fuselage
        cd0_fuselage = cf_fus * ff_fus * wet_area_fus / wing_area
        # Cockpit window (Gudmundsson p727)
        cd0_window = 0.002 * (height * width) / wing_area

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:fuselage:low_speed:CD0"] = cd0_fuselage + cd0_window
        else:
            outputs["data:aerodynamics:fuselage:cruise:CD0"] = cd0_fuselage + cd0_window

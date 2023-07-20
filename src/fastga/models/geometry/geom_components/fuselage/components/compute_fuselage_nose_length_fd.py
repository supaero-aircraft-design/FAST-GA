"""
Estimation of geometry of fuselage part A - Cabin (Commercial). 
"""

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

from ..constants import SUBMODEL_FUSELAGE_NOSE_LENGTH_FD


@oad.RegisterSubmodel(
    SUBMODEL_FUSELAGE_NOSE_LENGTH_FD,
    "fastga.submodel.geometry.fuselage.dimensions.nose_length.fd.legacy",
)
class ComputeFuselageNoseLengthFD(om.ExplicitComponent):
    """
    Computes nose length.
    """

    def setup(self):

        self.add_input("data:geometry:propulsion:nacelle:length", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:propeller:depth", val=np.nan, units="m")

        self.add_output("data:geometry:fuselage:front_length", units="m")

        self.declare_partials(
            "*",
            [
                "data:geometry:propulsion:nacelle:length",
                "data:geometry:propeller:depth",
                "data:geometry:fuselage:maximum_height",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        nacelle_length = inputs["data:geometry:propulsion:nacelle:length"]
        prop_layout = inputs["data:geometry:propulsion:engine:layout"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        spinner_length = inputs["data:geometry:propeller:depth"]

        if prop_layout == 3.0:  # engine located in nose
            lav = nacelle_length + spinner_length
        else:
            lav = 1.40 * h_f
            # Used to be 1.7, supposedly as an A320 according to FAST legacy. Results on the BE76
            # tend to say it is around 1.40, though it varies a lot depending on the airplane and
            # its use

        outputs["data:geometry:fuselage:front_length"] = lav

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        prop_layout = inputs["data:geometry:propulsion:engine:layout"]

        if prop_layout == 3.0:

            partials[
                "data:geometry:fuselage:front_length", "data:geometry:propulsion:nacelle:length"
            ] = 1.0
            partials["data:geometry:fuselage:front_length", "data:geometry:propeller:depth"] = 1.0
            partials[
                "data:geometry:fuselage:front_length", "data:geometry:fuselage:maximum_height"
            ] = 0.0

        else:

            partials[
                "data:geometry:fuselage:front_length", "data:geometry:propulsion:nacelle:length"
            ] = 0.0
            partials["data:geometry:fuselage:front_length", "data:geometry:propeller:depth"] = 0.0
            partials[
                "data:geometry:fuselage:front_length", "data:geometry:fuselage:maximum_height"
            ] = 1.40
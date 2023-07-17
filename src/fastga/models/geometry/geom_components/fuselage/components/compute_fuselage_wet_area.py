"""Estimation of fuselage wet area."""
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

from ..constants import SUBMODEL_FUSELAGE_WET_AREA

oad.RegisterSubmodel.active_models[
    SUBMODEL_FUSELAGE_WET_AREA
] = "fastga.submodel.geometry.fuselage.wet_area.legacy"


@oad.RegisterSubmodel(
    SUBMODEL_FUSELAGE_WET_AREA, "fastga.submodel.geometry.fuselage.wet_area.legacy"
)
class ComputeFuselageWetArea(om.ExplicitComponent):
    """
    Fuselage wet area estimation, based on a simple geometric description of the fuselage one
    cone at the front a cylinder in the middle and a cone at the back.
    """

    def setup(self):

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")

        self.add_output("data:geometry:fuselage:wet_area", units="m**2")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]

        # Using the simple geometric description
        fus_dia = np.sqrt(b_f * h_f)  # equivalent diameter of the fuselage
        cyl_length = fus_length - lav - lar
        wet_area_nose = 2.45 * fus_dia * lav
        wet_area_cyl = np.pi * fus_dia * cyl_length
        wet_area_tail = 2.3 * fus_dia * lar
        wet_area_fus = wet_area_nose + wet_area_cyl + wet_area_tail

        outputs["data:geometry:fuselage:wet_area"] = wet_area_fus

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]

        partials["data:geometry:fuselage:wet_area", "data:geometry:fuselage:maximum_width"] = (
            (23 * h_f * lar) / 20
            + (49 * h_f * lav) / 40
            - (h_f * np.pi * (lar - fus_length + lav)) / 2
        ) / (np.sqrt(b_f * h_f))
        partials["data:geometry:fuselage:wet_area", "data:geometry:fuselage:maximum_height"] = (
            (23 * b_f * lar) / 20
            + (49 * b_f * lav) / 40
            - (b_f * np.pi * (lar - fus_length + lav)) / 2
        ) / (np.sqrt(b_f * h_f))
        partials[
            "data:geometry:fuselage:wet_area", "data:geometry:fuselage:length"
        ] = np.pi * np.sqrt(b_f * h_f)
        partials["data:geometry:fuselage:wet_area", "data:geometry:fuselage:front_length"] = (
            49 / 20 - np.pi
        ) * np.sqrt(b_f * h_f)
        partials["data:geometry:fuselage:wet_area", "data:geometry:fuselage:rear_length"] = (
            23 / 10 - np.pi
        ) * np.sqrt(b_f * h_f)

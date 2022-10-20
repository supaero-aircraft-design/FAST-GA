"""Estimation of fuselage average depth around the vertical tail."""
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

from openmdao.core.explicitcomponent import ExplicitComponent

import fastoad.api as oad

from ..constants import SUBMODEL_FUSELAGE_VOLUME

oad.RegisterSubmodel.active_models[
    SUBMODEL_FUSELAGE_VOLUME
] = "fastga.submodel.geometry.fuselage.volume.legacy"


@oad.RegisterSubmodel(SUBMODEL_FUSELAGE_VOLUME, "fastga.submodel.geometry.fuselage.volume.legacy")
class ComputeFuselageVolume(ExplicitComponent):

    """
    Fuselage volume computation. Based on geometric consideration assuming the fuselage is
    cylindrical at the center section and conical at the front and back.
    """

    def setup(self):

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")

        self.add_output(
            "data:geometry:fuselage:volume",
            units="m**3",
            desc="Volume of the fuselage",
        )

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]
        lar = inputs["data:geometry:fuselage:rear_length"]
        lav = inputs["data:geometry:fuselage:front_length"]

        l_f = np.sqrt(b_f * h_f)
        l_cyc = fus_length - lav - lar
        # estimation of fuselage volume
        volume_fus = np.pi * l_f ** 2.0 / 4.0 * (0.7 * lav + 0.5 * lar + l_cyc)

        outputs["data:geometry:fuselage:volume"] = volume_fus

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]
        lar = inputs["data:geometry:fuselage:rear_length"]
        lav = inputs["data:geometry:fuselage:front_length"]

        partials["data:geometry:fuselage:volume", "data:geometry:fuselage:maximum_width"] = (
            np.pi * h_f / 4.0 * (-0.3 * lav - 0.5 * lar + fus_length)
        )
        partials["data:geometry:fuselage:volume", "data:geometry:fuselage:maximum_height"] = (
            np.pi * b_f / 4.0 * (-0.3 * lav - 0.5 * lar + fus_length)
        )
        partials["data:geometry:fuselage:volume", "data:geometry:fuselage:length"] = (
            np.pi * b_f * h_f / 4.0
        )
        partials["data:geometry:fuselage:volume", "data:geometry:fuselage:rear_length"] = (
            -0.5 * np.pi * b_f * h_f / 4.0
        )
        partials["data:geometry:fuselage:volume", "data:geometry:fuselage:front_length"] = (
            -0.3 * np.pi * b_f * h_f / 4.0
        )

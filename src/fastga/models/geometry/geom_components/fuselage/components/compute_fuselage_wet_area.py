"""
Python module for fuselage wet area calculation, part of the fuselage geometry.
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
import fastoad.api as oad

from ..constants import (
    SERVICE_FUSELAGE_WET_AREA,
    SUBMODEL_FUSELAGE_WET_AREA_LEGACY,
    SUBMODEL_FUSELAGE_WET_AREA_FLOPS,
)

oad.RegisterSubmodel.active_models[SERVICE_FUSELAGE_WET_AREA] = SUBMODEL_FUSELAGE_WET_AREA_LEGACY


@oad.RegisterSubmodel(SERVICE_FUSELAGE_WET_AREA, SUBMODEL_FUSELAGE_WET_AREA_LEGACY)
class ComputeFuselageWetArea(om.ExplicitComponent):
    """
    Fuselage wet area estimation, based on a simple geometric description of the fuselage one
    cone at the front a cylinder in the middle and a cone at the back, obtained from
    :cite:`supaero:2014`.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")

        self.add_output("data:geometry:fuselage:wet_area", units="m**2")

        self.declare_partials("data:geometry:fuselage:wet_area", "*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        cabin_length = inputs["data:geometry:cabin:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]

        # Using the simple geometric description
        fus_dia = np.sqrt(b_f * h_f)  # equivalent diameter of the fuselage
        wet_area_nose = 2.45 * fus_dia * lav
        wet_area_cyl = np.pi * fus_dia * cabin_length
        wet_area_tail = 2.3 * fus_dia * lar
        wet_area_fus = wet_area_nose + wet_area_cyl + wet_area_tail

        outputs["data:geometry:fuselage:wet_area"] = wet_area_fus

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        cabin_length = inputs["data:geometry:cabin:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]
        fus_dia = np.sqrt(b_f * h_f)

        partials["data:geometry:fuselage:wet_area", "data:geometry:fuselage:maximum_width"] = (
            1.225 * h_f * lav + 0.5 * (h_f * np.pi * cabin_length) + 1.15 * h_f * lar
        ) / fus_dia
        partials["data:geometry:fuselage:wet_area", "data:geometry:fuselage:maximum_height"] = (
            1.225 * b_f * lav + 0.5 * (b_f * np.pi * cabin_length) + 1.15 * b_f * lar
        ) / fus_dia
        partials["data:geometry:fuselage:wet_area", "data:geometry:cabin:length"] = np.pi * fus_dia
        partials["data:geometry:fuselage:wet_area", "data:geometry:fuselage:front_length"] = (
            2.45 * fus_dia
        )
        partials["data:geometry:fuselage:wet_area", "data:geometry:fuselage:rear_length"] = (
            2.3 * fus_dia
        )


@oad.RegisterSubmodel(SERVICE_FUSELAGE_WET_AREA, SUBMODEL_FUSELAGE_WET_AREA_FLOPS)
class ComputeFuselageWetAreaFLOPS(om.ExplicitComponent):
    """
    Fuselage wet area estimation, determined based on Equation 61 from :cite:`wells:2017`.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")

        self.add_output("data:geometry:fuselage:wet_area", units="m**2")

        self.declare_partials("data:geometry:fuselage:wet_area", "*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]

        # Using the formula from The Flight Optimization System Weights Estimation Method
        fus_dia = np.sqrt(b_f * h_f)  # equivalent diameter of the fuselage
        wet_area_fus = np.pi * (fus_length / fus_dia - 1.7) * fus_dia**2.0

        outputs["data:geometry:fuselage:wet_area"] = wet_area_fus

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]
        fus_dia = np.sqrt(b_f * h_f)

        partials["data:geometry:fuselage:wet_area", "data:geometry:fuselage:maximum_width"] = (
            h_f * np.pi * (fus_length / fus_dia - 1.7)
            - (np.pi * b_f * fus_length * h_f * 2.0) / (2.0 * fus_dia**3.0)
        )
        partials["data:geometry:fuselage:wet_area", "data:geometry:fuselage:maximum_height"] = (
            b_f * np.pi * (fus_length / fus_dia - 1.7)
            - (np.pi * b_f**2.0 * fus_length * h_f) / (2.0 * fus_dia**3.0)
        )
        partials["data:geometry:fuselage:wet_area", "data:geometry:fuselage:length"] = (
            np.pi * b_f * h_f
        ) / fus_dia

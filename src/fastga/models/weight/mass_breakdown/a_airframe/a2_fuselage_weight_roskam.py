"""Estimation of fuselage weight."""
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

import logging

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from .constants import SUBMODEL_FUSELAGE_MASS

_LOGGER = logging.getLogger(__name__)


@oad.RegisterSubmodel(
    SUBMODEL_FUSELAGE_MASS, "fastga.submodel.weight.mass.airframe.fuselage.roskam"
)
class ComputeFuselageWeightRoskam(om.ExplicitComponent):
    """
    Fuselage weight estimation, includes the computation of the fuselage weight for a high wing
    aircraft.

    Based on : Roskam. Airplane design - Part 5: component weight estimation

    """

    def setup(self):

        self.add_input("data:geometry:fuselage:length", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="ft")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="ft")
        self.add_input("data:geometry:wing_configuration", val=np.nan)

        self.add_output("data:weight:airframe:fuselage:mass", units="lb")

        self.declare_partials(
            of="*",
            wrt=[
                "data:geometry:fuselage:length",
                "data:geometry:fuselage:front_length",
                "data:weight:aircraft:MTOW",
                "data:geometry:cabin:seats:passenger:NPAX_max",
                "data:geometry:fuselage:maximum_width",
                "data:geometry:fuselage:maximum_width",
                "data:geometry:fuselage:maximum_height",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        maximum_width = inputs["data:geometry:fuselage:maximum_width"]
        maximum_height = inputs["data:geometry:fuselage:maximum_height"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        npax_max = (
            inputs["data:geometry:cabin:seats:passenger:NPAX_max"] + 2.0
        )  # addition of 2 pilots
        wing_config = inputs["data:geometry:wing_configuration"]

        fus_dia = (maximum_height + maximum_width) / 2.0
        p_max = 2 * np.pi * (fus_dia / 2)  # maximum perimeter of the fuselage

        if wing_config == 1.0:

            # The formula found in Roskam originally contains a division by 100, but it leads to
            # results way too low. It will be omitted here. It does not seem to cause an issue
            # for the high wing configuration however, so we will simply issue a warning with a
            # recommendation to switch method for low wing aircraft
            a2 = 0.04682 * (mtow ** 0.692 * npax_max ** 0.374 * (fus_length - lav) ** 0.590)

            _LOGGER.warning(
                "This submodel is not trusted for the computation of the fuselage weight of low "
                "wing aircraft as it gives very small results. Consider switching submodel"
            )

        elif wing_config == 3.0:

            a2 = 14.86 * (
                mtow ** 0.144
                * ((fus_length - lav) / p_max) ** 0.778
                * (fus_length - lav) ** 0.383
                * npax_max ** 0.455
            )

        else:
            _LOGGER.info(
                "No formula available for the weight of the fuselage with a mid-wing "
                "configuration, taking high wing instead"
            )
            a2 = 14.86 * (
                mtow ** 0.144
                * ((fus_length - lav) / p_max) ** 0.778
                * (fus_length - lav) ** 0.383
                * npax_max ** 0.455
            )

        outputs["data:weight:airframe:fuselage:mass"] = a2

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        wing_config = inputs["data:geometry:wing_configuration"]

        if wing_config == 1.0:

            fus_length = inputs["data:geometry:fuselage:length"]
            lav = inputs["data:geometry:fuselage:front_length"]
            mtow = inputs["data:weight:aircraft:MTOW"]
            npax_max = (
                inputs["data:geometry:cabin:seats:passenger:NPAX_max"] + 2.0
            )  # addition of 2 pilots

            partials[
                "data:weight:airframe:fuselage:mass", "data:weight:aircraft:MTOW"
            ] = 0.04682 * (0.692 * mtow ** -0.308 * npax_max ** 0.374 * (fus_length - lav) ** 0.590)
            partials[
                "data:weight:airframe:fuselage:mass", "data:geometry:cabin:seats:passenger:NPAX_max"
            ] = 0.04682 * (mtow ** 0.692 * 0.374 * npax_max ** -0.626 * (fus_length - lav) ** 0.590)
            partials[
                "data:weight:airframe:fuselage:mass", "data:geometry:fuselage:length"
            ] = 0.04682 * (mtow ** 0.692 * npax_max ** 0.374 * 0.590 * (fus_length - lav) ** -0.41)
            partials[
                "data:weight:airframe:fuselage:mass", "data:geometry:fuselage:front_length"
            ] = -(
                0.04682 * (mtow ** 0.692 * npax_max ** 0.374 * 0.590 * (fus_length - lav) ** -0.41)
            )
            partials[
                "data:weight:airframe:fuselage:mass", "data:geometry:fuselage:maximum_width"
            ] = 0.0
            partials[
                "data:weight:airframe:fuselage:mass", "data:geometry:fuselage:maximum_height"
            ] = 0.0

        else:

            fus_length = inputs["data:geometry:fuselage:length"]
            lav = inputs["data:geometry:fuselage:front_length"]
            mtow = inputs["data:weight:aircraft:MTOW"]
            npax_max = (
                inputs["data:geometry:cabin:seats:passenger:NPAX_max"] + 2.0
            )  # addition of 2 pilots

            maximum_width = inputs["data:geometry:fuselage:maximum_width"]
            maximum_height = inputs["data:geometry:fuselage:maximum_height"]

            p_max = (
                np.pi * (maximum_height + maximum_width) / 2
            )  # maximum perimeter of the fuselage

            partials["data:weight:airframe:fuselage:mass", "data:weight:aircraft:MTOW"] = 14.86 * (
                0.144
                * mtow ** -0.856
                * ((fus_length - lav) / p_max) ** 0.778
                * (fus_length - lav) ** 0.383
                * npax_max ** 0.455
            )
            partials[
                "data:weight:airframe:fuselage:mass", "data:geometry:cabin:seats:passenger:NPAX_max"
            ] = 14.86 * (
                mtow ** 0.144
                * ((fus_length - lav) / p_max) ** 0.778
                * (fus_length - lav) ** 0.383
                * 0.455
                * npax_max ** -0.545
            )
            partials[
                "data:weight:airframe:fuselage:mass", "data:geometry:fuselage:length"
            ] = 14.86 * (
                mtow ** 0.144
                * p_max ** -0.778
                * 1.161
                * (fus_length - lav) ** 0.161
                * npax_max ** 0.455
            )
            partials[
                "data:weight:airframe:fuselage:mass", "data:geometry:fuselage:front_length"
            ] = -14.86 * (
                mtow ** 0.144
                * p_max ** -0.778
                * 1.161
                * (fus_length - lav) ** 0.161
                * npax_max ** 0.455
            )
            partials[
                "data:weight:airframe:fuselage:mass", "data:geometry:fuselage:maximum_width"
            ] = (
                14.86
                * (
                    mtow ** 0.144
                    * -0.778
                    * p_max ** -1.778
                    * (fus_length - lav) ** 1.161
                    * npax_max ** 0.455
                )
                * np.pi
                / 2.0
            )
            partials[
                "data:weight:airframe:fuselage:mass", "data:geometry:fuselage:maximum_height"
            ] = (
                14.86
                * (
                    mtow ** 0.144
                    * -0.778
                    * p_max ** -1.778
                    * (fus_length - lav) ** 1.161
                    * npax_max ** 0.455
                )
                * np.pi
                / 2.0
            )

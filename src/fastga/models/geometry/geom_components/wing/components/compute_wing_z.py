"""Estimation of wing Zs."""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2023  ONERA & ISAE-SUPAERO
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

from fastga.models.geometry.geom_components.wing.constants import SUBMODEL_WING_HEIGHT

_LOGGER = logging.getLogger(__name__)


@oad.RegisterSubmodel(SUBMODEL_WING_HEIGHT, "fastga.submodel.geometry.wing.height.legacy")
class ComputeWingZ(om.ExplicitComponent):
    """
    Computation of the distance between the fuselage center line and the wing. Based on simple
    geometric considerations.
    """

    def setup(self):

        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:dihedral", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:wing_configuration", val=np.nan)

        self.add_output(
            "data:geometry:wing:root:z",
            units="m",
            desc="Distance between the wing aerodynamic center at the root and the fuselage "
            "centerline, taken positive when wing is below the fuselage centerline",
        )
        self.add_output(
            "data:geometry:wing:tip:z",
            units="m",
            desc="Distance between the wing aerodynamic center at the tip and the fuselage "
            "centerline, taken positive when wing is below the fuselage centerline",
        )

        self.declare_partials(
            of="data:geometry:wing:root:z",
            wrt=[
                "data:geometry:wing:root:thickness_ratio",
                "data:geometry:wing:root:chord",
                "data:geometry:fuselage:maximum_height",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:geometry:wing:tip:z",
            wrt=[
                "data:geometry:wing:root:y",
                "data:geometry:wing:tip:y",
                "data:geometry:wing:tip:thickness_ratio",
                "data:geometry:wing:tip:chord",
                "data:geometry:wing:dihedral",
                "data:geometry:fuselage:maximum_height",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        dihedral_angle = inputs["data:geometry:wing:dihedral"]
        root_thickness_ratio = inputs["data:geometry:wing:root:thickness_ratio"]
        tip_thickness_ratio = inputs["data:geometry:wing:tip:thickness_ratio"]
        fus_height = inputs["data:geometry:fuselage:maximum_height"]
        wing_config = inputs["data:geometry:wing_configuration"]

        # Convention is positive in a low wing configuration and negative otherwise, see Roskam
        # part VI page 384 in the graph description

        if wing_config == 1.0:

            z2_wing = 0.5 * fus_height - 0.5 * root_thickness_ratio * l2_wing
            z4_wing = (
                0.5 * fus_height
                - (y4_wing - y2_wing) * np.tan(dihedral_angle)
                - 0.5 * tip_thickness_ratio * l4_wing
            )
            # Positive dihedral reduce distance between wing AC and fuselage centerline

        elif wing_config == 2.0:

            # For mid-wing configuration the root AC is at the same height as the fuselage
            # centerline

            z2_wing = 0.0
            z4_wing = -(y4_wing - y2_wing) * np.tan(dihedral_angle)

        elif wing_config == 3.0:

            z2_wing = -0.5 * fus_height + 0.5 * root_thickness_ratio * l2_wing
            z4_wing = (
                -0.5 * fus_height
                - (y4_wing - y2_wing) * np.tan(dihedral_angle)
                + 0.5 * tip_thickness_ratio * l4_wing
            )

        else:
            _LOGGER.warning(
                "Wing configuration %s unknown, replaced by low wing configuration", wing_config
            )
            z2_wing = 0.5 * fus_height - 0.5 * root_thickness_ratio * l2_wing
            z4_wing = (
                0.5 * fus_height
                - (y4_wing - y2_wing) * np.tan(dihedral_angle)
                - 0.5 * tip_thickness_ratio * l4_wing
            )

        outputs["data:geometry:wing:root:z"] = z2_wing
        outputs["data:geometry:wing:tip:z"] = z4_wing

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        root_thickness_ratio = inputs["data:geometry:wing:root:thickness_ratio"]
        tip_thickness_ratio = inputs["data:geometry:wing:tip:thickness_ratio"]
        wing_config = inputs["data:geometry:wing_configuration"]

        dihedral_angle = inputs["data:geometry:wing:dihedral"]

        if wing_config == 2.0:

            partials["data:geometry:wing:root:z", "data:geometry:wing:root:thickness_ratio"] = 0.0
            partials["data:geometry:wing:root:z", "data:geometry:wing:root:chord"] = 0.0
            partials["data:geometry:wing:root:z", "data:geometry:fuselage:maximum_height"] = 0.0

            partials["data:geometry:wing:tip:z", "data:geometry:wing:root:y"] = -np.tan(
                dihedral_angle
            )
            partials["data:geometry:wing:tip:z", "data:geometry:wing:tip:y"] = np.tan(
                dihedral_angle
            )
            partials["data:geometry:wing:tip:z", "data:geometry:wing:tip:thickness_ratio"] = 0.0
            partials["data:geometry:wing:tip:z", "data:geometry:wing:tip:chord"] = 0.0
            partials["data:geometry:wing:tip:z", "data:geometry:wing:dihedral"] = -(
                y4_wing - y2_wing
            )
            partials["data:geometry:wing:tip:z", "data:geometry:fuselage:maximum_height"] = 0.0

        elif wing_config == 3.0:

            partials["data:geometry:wing:root:z", "data:geometry:wing:root:thickness_ratio"] = (
                0.5 * l2_wing
            )
            partials["data:geometry:wing:root:z", "data:geometry:wing:root:chord"] = (
                0.5 * root_thickness_ratio
            )
            partials["data:geometry:wing:root:z", "data:geometry:fuselage:maximum_height"] = -0.5

            partials["data:geometry:wing:tip:z", "data:geometry:wing:root:y"] = np.tan(
                dihedral_angle
            )
            partials["data:geometry:wing:tip:z", "data:geometry:wing:tip:y"] = -np.tan(
                dihedral_angle
            )
            partials["data:geometry:wing:tip:z", "data:geometry:wing:tip:thickness_ratio"] = (
                0.5 * l4_wing
            )
            partials["data:geometry:wing:tip:z", "data:geometry:wing:tip:chord"] = (
                0.5 * tip_thickness_ratio
            )
            partials["data:geometry:wing:tip:z", "data:geometry:wing:dihedral"] = -(
                y4_wing - y2_wing
            )
            partials["data:geometry:wing:tip:z", "data:geometry:fuselage:maximum_height"] = -0.5

        else:

            partials["data:geometry:wing:root:z", "data:geometry:wing:root:thickness_ratio"] = (
                -0.5 * l2_wing
            )
            partials["data:geometry:wing:root:z", "data:geometry:wing:root:chord"] = (
                -0.5 * root_thickness_ratio
            )
            partials["data:geometry:wing:root:z", "data:geometry:fuselage:maximum_height"] = 0.5

            partials["data:geometry:wing:tip:z", "data:geometry:wing:root:y"] = np.tan(
                dihedral_angle
            )
            partials["data:geometry:wing:tip:z", "data:geometry:wing:tip:y"] = -np.tan(
                dihedral_angle
            )
            partials["data:geometry:wing:tip:z", "data:geometry:wing:tip:thickness_ratio"] = (
                -0.5 * l4_wing
            )
            partials["data:geometry:wing:tip:z", "data:geometry:wing:tip:chord"] = (
                -0.5 * tip_thickness_ratio
            )
            partials["data:geometry:wing:tip:z", "data:geometry:wing:dihedral"] = -(
                y4_wing - y2_wing
            )
            partials["data:geometry:wing:tip:z", "data:geometry:fuselage:maximum_height"] = 0.5

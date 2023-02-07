"""Estimation of wing Zs."""
# This submodel estimates the vertical wing position of the aircraft
# taking into account the dihedral angle of the wing as well as it's wing location(high,mid,low)
# 1.0 - low wing, 2.0 - mid wing, 3.0 - high wing

import numpy as np

import math

from openmdao.core.explicitcomponent import ExplicitComponent

import fastoad.api as oad

import logging

from fastga.models.geometry.geom_components.wing.constants import SUBMODEL_WING_LOCATION

_LOGGER = logging.getLogger(__name__)


@oad.RegisterSubmodel(SUBMODEL_WING_LOCATION, "fastga.submodel.geometry.wing.location.legacy2")
class ComputeWingZ(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Wing Zs estimation."""

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

        self.add_output("data:geometry:wing:root:z_lower", units="m")
        self.add_output("data:geometry:wing:root:z_upper", units="m")
        self.add_output("data:geometry:wing:tip:z_lower", units="m")
        self.add_output("data:geometry:wing:tip:z_upper", units="m")

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

        if wing_config == 1.0:

            z2_lower = 0.25 * fus_height
            z2_upper = 0.25 * fus_height + root_thickness_ratio * l2_wing
            z4_lower = z2_lower + (y4_wing - y2_wing) * math.tan(dihedral_angle)
            z4_upper = z4_lower + tip_thickness_ratio * l4_wing

            if dihedral_angle < 0.0:
                _LOGGER.warning(
                    "You have chosen a negative dihedral wing for a low wing configuration"
                )

        elif wing_config == 2.0:

            z2_lower = 0.50 * fus_height
            z2_upper = 0.50 * fus_height + root_thickness_ratio * l2_wing
            z4_lower = z2_lower + (y4_wing - y2_wing) * math.tan(dihedral_angle)
            z4_upper = z4_lower + tip_thickness_ratio * l4_wing

            if dihedral_angle < 0.0:
                _LOGGER.warning(
                    "You have chosen a negative dihedral wing for a mid wing configuration"
                )

        elif wing_config == 3.0:

            z2_lower = 1.0 * fus_height
            z2_upper = 1.0 * fus_height + root_thickness_ratio * l2_wing
            z4_lower = z2_lower + (y4_wing - y2_wing) * math.tan(dihedral_angle)
            z4_upper = z4_lower + tip_thickness_ratio * l4_wing

        outputs["data:geometry:wing:root:z_lower"] = z2_lower
        outputs["data:geometry:wing:root:z_upper"] = z2_upper
        outputs["data:geometry:wing:tip:z_lower"] = z4_lower
        outputs["data:geometry:wing:tip:z_upper"] = z4_upper

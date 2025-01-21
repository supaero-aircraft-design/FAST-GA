"""
Computation of wing area update and constraints based on the amount of fuel in the wing with
advanced computation of the maximum fuel weight.
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

import logging

import fastoad.api as oad
import numpy as np
import openmdao.api as om

from fastga.models.geometry.geom_components.wing.components.compute_wing_l1_l4 import (
    ComputeWingL1AndL4,
)
from fastga.models.geometry.geom_components.wing.components.compute_wing_l2_l3 import (
    ComputeWingL2AndL3,
)
from fastga.models.geometry.geom_components.wing.components.compute_wing_y import (
    ComputeWingY,
)
from fastga.models.geometry.geom_components.wing_tank.compute_mfw_advanced import (
    ComputeMFWAdvanced,
)
from ..constants import SUBMODEL_WING_AREA_GEOM_LOOP, SUBMODEL_WING_AREA_GEOM_CONS

_LOGGER = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
# Overriding only necessary OpenMDAO methods
class UpdateMFW(om.Group):
    """
    Computes the MFW for a given value of the wing area by reusing only necessary components
    """

    def setup(self):
        self.add_subsystem(
            name="update_wing_y",
            subsys=ComputeWingY(),
            promotes_inputs=[
                "data:geometry:wing:aspect_ratio",
                "data:geometry:fuselage:maximum_width",
                "data:geometry:wing:kink:span_ratio",
                ("data:geometry:wing:area", "wing_area"),
            ],
            promotes_outputs=[],
        )
        self.add_subsystem(
            name="update_wing_l2_l3",
            subsys=ComputeWingL2AndL3(),
            promotes_inputs=[
                "data:geometry:wing:taper_ratio",
                ("data:geometry:wing:area", "wing_area"),
            ],
            promotes_outputs=[],
        )
        self.add_subsystem(
            name="update_wing_l1_l4",
            subsys=ComputeWingL1AndL4(),
            promotes_inputs=[
                "data:geometry:wing:taper_ratio",
                ("data:geometry:wing:area", "wing_area"),
            ],
            promotes_outputs=[],
        )
        self.add_subsystem(
            name="update_wing_mfw",
            subsys=ComputeMFWAdvanced(),
            promotes_outputs=["*"],
            promotes_inputs=[
                "data:propulsion:fuel_type",
                "data:geometry:wing:root:thickness_ratio",
                "data:geometry:wing:tip:thickness_ratio",
                "data:geometry:flap:chord_ratio",
                "data:geometry:wing:aileron:chord_ratio",
                "data:geometry:propulsion:tank:y_ratio_tank_beginning",
                "data:geometry:propulsion:tank:y_ratio_tank_end",
                "data:geometry:propulsion:engine:layout",
                "data:geometry:propulsion:engine:y_ratio",
                "data:geometry:propulsion:tank:LE_chord_percentage",
                "data:geometry:propulsion:tank:TE_chord_percentage",
                "data:geometry:propulsion:nacelle:width",
                "data:geometry:landing_gear:type",
                "data:geometry:landing_gear:y",
                "settings:geometry:fuel_tanks:depth",
            ],
        )

        self.connect(
            "update_wing_y.data:geometry:wing:span",
            "update_wing_mfw.data:geometry:wing:span",
        )
        self.connect(
            "update_wing_y.data:geometry:wing:root:y",
            [
                "update_wing_mfw.data:geometry:wing:root:y",
                "update_wing_l2_l3.data:geometry:wing:root:y",
                "update_wing_l1_l4.data:geometry:wing:root:y",
            ],
        )
        self.connect(
            "update_wing_y.data:geometry:wing:tip:y",
            [
                "update_wing_mfw.data:geometry:wing:tip:y",
                "update_wing_l2_l3.data:geometry:wing:tip:y",
                "update_wing_l1_l4.data:geometry:wing:tip:y",
            ],
        )

        self.connect(
            "update_wing_l2_l3.data:geometry:wing:root:chord",
            "update_wing_mfw.data:geometry:wing:root:chord",
        )

        self.connect(
            "update_wing_l1_l4.data:geometry:wing:tip:chord",
            "update_wing_mfw.data:geometry:wing:tip:chord",
        )


class DistanceToMFWForUpdate(om.ImplicitComponent):
    """
    ImplicitComponent that readjust the value of the wing area depending on how far from the
    target (fuel for sizing mission) the MFW is.
    """

    def setup(self):
        self.add_input("data:mission:sizing:fuel", units="kg", val=np.nan)
        self.add_input("data:weight:aircraft:MFW", units="kg", val=np.nan)

        self.add_output("wing_area", val=15.0, units="m**2")

        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument, too-many-arguments, too-many-positional-arguments
    # Overriding OpenMDAO apply_nonlinear, not all arguments are used
    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        # Wing area will likely be in tens of m2 while MFW will likely be in hundreds of
        residuals["wing_area"] = (
            inputs["data:weight:aircraft:MFW"] - inputs["data:mission:sizing:fuel"]
        ) / 10.0

    # pylint: disable=missing-function-docstring, too-many-arguments, unused-argument, too-many-positional-arguments
    # Overriding OpenMDAO linearize, not all arguments are used
    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):
        jacobian["wing_area", "data:weight:aircraft:MFW"] = 0.1
        jacobian["wing_area", "data:mission:sizing:fuel"] = -0.1


# pylint: disable=too-few-public-methods
# Overriding only necessary OpenMDAO methods
@oad.RegisterSubmodel(
    SUBMODEL_WING_AREA_GEOM_LOOP, "fastga.submodel.loop.wing_area.update.geom.advanced"
)
class UpdateWingAreaGeomAdvanced(om.Group):
    """
    Group that iterates on the wing area to find the value which equates the max fuel weight to the
    fuel necessary for the mission.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Solvers setup
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 15
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.nonlinear_solver.options["atol"] = 1e-2
        self.nonlinear_solver.options["stall_limit"] = 5
        self.nonlinear_solver.options["stall_tol"] = 1e-5
        self.linear_solver = om.DirectSolver()

    def setup(self):
        self.add_subsystem(
            name="update_area",
            subsys=DistanceToMFWForUpdate(),
            promotes_inputs=["data:mission:sizing:fuel"],
            promotes_outputs=["wing_area"],
        )
        self.add_subsystem(name="update_mfw", subsys=UpdateMFW(), promotes_inputs=["*"])

        self.connect(
            "update_mfw.data:weight:aircraft:MFW",
            "update_area.data:weight:aircraft:MFW",
        )


class DistanceToMFWForConstraint(om.ExplicitComponent):
    """
    Computation of the distance to the geometric constraints which translates as additional fuel
    capacity.
    """

    def setup(self):
        self.add_input("data:mission:sizing:fuel", units="kg", val=np.nan)
        self.add_input("MFW", units="kg", val=np.nan)

        self.add_output("data:constraints:wing:additional_fuel_capacity", val=0.0, units="kg")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:constraints:wing:additional_fuel_capacity"] = (
            inputs["MFW"] - inputs["data:mission:sizing:fuel"]
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["data:constraints:wing:additional_fuel_capacity", "MFW"] = 1.0
        partials[
            "data:constraints:wing:additional_fuel_capacity", "data:mission:sizing:fuel"
        ] = -1.0


# pylint: disable=too-few-public-methods
# Overriding only necessary OpenMDAO methods
@oad.RegisterSubmodel(
    SUBMODEL_WING_AREA_GEOM_CONS,
    "fastga.submodel.loop.wing_area.constraint.geom.advanced",
)
class ConstraintWingAreaGeomAdvanced(om.Group):
    """
    Computes the difference between what the wing can store in terms of fuel and the fuel
    needed for the mission to check if the constraints is respected using an advanced geometric
    computation of the fuel inside the wing.
    """

    def setup(self):
        # To rename the wing area as it is used in the UpdateMFW() group
        area_rename = om.AddSubtractComp()
        area_rename.add_equation(
            "wing_area",
            [
                "data:geometry:wing:area",
                "data:geometry:wing:area",
            ],
            scaling_factors=[0.5, 0.5],
            units="m**2",
        )

        self.add_subsystem("wing_area_rename", area_rename, promotes=["data:*"])
        self.add_subsystem(
            name="update_mfw",
            subsys=UpdateMFW(),
            promotes_inputs=["data:*", "settings:*"],
            promotes_outputs=[],
        )
        self.add_subsystem(
            name="compute_constraints_area",
            subsys=DistanceToMFWForConstraint(),
            promotes=["data:*"],
        )

        self.connect("wing_area_rename.wing_area", "update_mfw.wing_area")
        self.connect("update_mfw.data:weight:aircraft:MFW", "compute_constraints_area.MFW")

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
import copy

import numpy as np
import openmdao.api as om
from openmdao.utils.units import convert_units

from scipy.optimize import fsolve

import fastoad.api as oad

from fastga.command.api import generate_block_analysis

from fastga.models.geometry.geom_components.wing.components.compute_wing_y import ComputeWingY
from fastga.models.geometry.geom_components.wing.components.compute_wing_l2_l3 import (
    ComputeWingL2AndL3,
)
from fastga.models.geometry.geom_components.wing.components.compute_wing_l1_l4 import (
    ComputeWingL1AndL4,
)
from fastga.models.geometry.geom_components.wing_tank.compute_mfw_advanced import ComputeMFWAdvanced

from ..constants import SUBMODEL_WING_AREA_GEOM_LOOP, SUBMODEL_WING_AREA_GEOM_CONS

_LOGGER = logging.getLogger(__name__)


@oad.RegisterSubmodel(
    SUBMODEL_WING_AREA_GEOM_LOOP, "fastga.submodel.loop.wing_area.update.geom.advanced"
)
class UpdateWingAreaGeomAdvanced(om.ExplicitComponent):
    """
    Computes needed wing area to be able to load enough fuel to achieve the sizing mission. For
    the mission wing area the code uses the fsolve algorithm on a function that computes the wing
    area following the same approach as in compute_mfw.
    """

    def setup(self):

        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:kink:span_ratio", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:mission:sizing:fuel", val=np.nan, units="kg")
        self.add_input("data:propulsion:fuel_type", val=np.nan)
        self.add_input("data:geometry:propulsion:nacelle:width", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:type", val=np.nan)
        self.add_input("data:geometry:landing_gear:y", val=np.nan, units="m")
        self.add_input("settings:geometry:fuel_tanks:depth", val=np.nan)
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input(
            "data:geometry:propulsion:engine:y_ratio",
            shape_by_conn=True,
        )
        self.add_input("data:geometry:propulsion:tank:LE_chord_percentage", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:TE_chord_percentage", val=np.nan)
        self.add_input("data:geometry:flap:chord_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aileron:chord_ratio", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:y_ratio_tank_beginning", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:y_ratio_tank_end", val=np.nan)

        self.add_output("wing_area", val=10.0, units="m**2")

        self.declare_partials(
            "*",
            "*",
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mfw_mission = inputs["data:mission:sizing:fuel"]

        # TODO: Why is it hardcoded ?
        wing_area_mission_initial = 16.8871

        wing_area_mission, _, ier, _ = fsolve(
            compute_wing_area_new,
            np.array([wing_area_mission_initial]),
            args=(inputs, mfw_mission),
            xtol=0.01,
            full_output=True,
        )

        if ier != 1:
            _LOGGER.warning(
                "Could not find a wing area that suits the requirement for fuel inside the wing, "
                "setting the value to 0.0",
            )

        outputs["wing_area"] = wing_area_mission


@oad.RegisterSubmodel(
    SUBMODEL_WING_AREA_GEOM_CONS, "fastga.submodel.loop.wing_area.constraint.geom.advanced"
)
class ConstraintWingAreaGeomAdvanced(om.ExplicitComponent):
    """
    Computes the difference between what the wing can store in terms of fuel and the fuel
    needed for the mission to check if the constraints is respected using an advanced geometric
    computation of the fuel inside the wing.
    """

    def setup(self):
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:kink:span_ratio", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:mission:sizing:fuel", val=np.nan, units="kg")
        self.add_input("data:propulsion:fuel_type", val=np.nan)
        self.add_input("data:geometry:propulsion:nacelle:width", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:type", val=np.nan)
        self.add_input("data:geometry:landing_gear:y", val=np.nan, units="m")
        self.add_input("settings:geometry:fuel_tanks:depth", val=np.nan)
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input(
            "data:geometry:propulsion:engine:y_ratio",
            shape_by_conn=True,
        )
        self.add_input("data:geometry:propulsion:tank:LE_chord_percentage", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:TE_chord_percentage", val=np.nan)
        self.add_input("data:geometry:flap:chord_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aileron:chord_ratio", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:y_ratio_tank_beginning", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:y_ratio_tank_end", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("data:constraints:wing:additional_fuel_capacity", units="kg")

        self.declare_partials(
            "*",
            "*",
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mfw_mission = inputs["data:mission:sizing:fuel"]
        wing_area = inputs["data:geometry:wing:area"]

        outputs["data:constraints:wing:additional_fuel_capacity"] = compute_wing_area_new(
            wing_area, inputs, mfw_mission
        )


def compute_wing_area_new(wing_area, inputs, fuel_mission):
    """
    Computes the difference between the mfw for a given wing area and the fuel that we need to
    store inside the mission, when solved, there is just enough room inside the wing to hold the
    fuel

    :param wing_area: wing area, in m**2
    :param inputs: inputs of the component
    :param fuel_mission: fuel needed to achieve the mission, in kg
    """
    wing_ar = float(inputs["data:geometry:wing:aspect_ratio"])
    wing_taper_ratio = float(inputs["data:geometry:wing:taper_ratio"])
    fus_width = float(inputs["data:geometry:fuselage:maximum_width"])
    kink_span_ratio = float(inputs["data:geometry:wing:kink:span_ratio"])

    fuel_type = float(inputs["data:propulsion:fuel_type"])
    root_tc = float(inputs["data:geometry:wing:root:thickness_ratio"])
    tip_tc = float(inputs["data:geometry:wing:tip:thickness_ratio"])
    flap_chord_ratio = float(inputs["data:geometry:flap:chord_ratio"])
    aileron_chord_ratio = float(inputs["data:geometry:wing:aileron:chord_ratio"])
    y_ratio_tank_beginning = float(inputs["data:geometry:propulsion:tank:y_ratio_tank_beginning"])
    y_ratio_tank_end = float(inputs["data:geometry:propulsion:tank:y_ratio_tank_end"])
    engine_config = float(inputs["data:geometry:propulsion:engine:layout"])
    y_ratio_tank = inputs["data:geometry:propulsion:engine:y_ratio"]
    le_chord_percentage = float(inputs["data:geometry:propulsion:tank:LE_chord_percentage"])
    te_chord_percentage = float(inputs["data:geometry:propulsion:tank:TE_chord_percentage"])
    nacelle_width = float(inputs["data:geometry:propulsion:nacelle:width"])
    lg_type = float(inputs["data:geometry:landing_gear:type"])
    y_lg = float(inputs["data:geometry:landing_gear:y"])
    k = float(inputs["settings:geometry:fuel_tanks:depth"])

    # We first have to recompute all the data needed for the tank capacity computation that
    # depends on the wing area To ensure coherency with the method used, we will use the
    # generate block analysis method on the component that compute said value and then use a
    # block_analysis on the compute_mfw_advanced component.

    # First we need to compute the y positions and the span

    var_inputs_compute_y = [
        "data:geometry:wing:aspect_ratio",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:wing:area",
        "data:geometry:wing:kink:span_ratio",
    ]

    compute_wing_y = generate_block_analysis(ComputeWingY(), var_inputs_compute_y, "", False)

    var_dict_compute_y = {
        "data:geometry:wing:aspect_ratio": (wing_ar, None),
        "data:geometry:fuselage:maximum_width": (fus_width, "m"),
        "data:geometry:wing:area": (wing_area, "m**2"),
        "data:geometry:wing:kink:span_ratio": (kink_span_ratio, None),
    }

    var_outputs_compute_y = compute_wing_y(var_dict_compute_y)

    # We ensure that the units are in the SI system using openmdao convert unit_function
    wing_span_original_val = var_outputs_compute_y.get("data:geometry:wing:span")[0]
    wing_span_original_unit = var_outputs_compute_y.get("data:geometry:wing:span")[1]
    wing_span = convert_units(wing_span_original_val, wing_span_original_unit, "m")

    root_y_original_val = var_outputs_compute_y.get("data:geometry:wing:root:y")[0]
    root_y_original_unit = var_outputs_compute_y.get("data:geometry:wing:root:y")[1]
    root_y = convert_units(root_y_original_val, root_y_original_unit, "m")

    tip_y_original_val = var_outputs_compute_y.get("data:geometry:wing:tip:y")[0]
    tip_y_original_unit = var_outputs_compute_y.get("data:geometry:wing:tip:y")[1]
    tip_y = convert_units(tip_y_original_val, tip_y_original_unit, "m")

    # We now compute the root chord

    var_inputs_compute_root_chord = [
        "data:geometry:wing:taper_ratio",
        "data:geometry:wing:area",
        "data:geometry:wing:root:y",
        "data:geometry:wing:tip:y",
    ]

    compute_wing_root_chord = generate_block_analysis(
        ComputeWingL2AndL3(), var_inputs_compute_root_chord, "", False
    )

    var_dict_compute_root_chord = {
        "data:geometry:wing:taper_ratio": (wing_taper_ratio, None),
        "data:geometry:wing:area": (wing_area, "m**2"),
        "data:geometry:wing:root:y": (root_y, "m"),
        "data:geometry:wing:tip:y": (tip_y, "m"),
    }

    var_outputs_compute_root_chord = compute_wing_root_chord(var_dict_compute_root_chord)

    # We ensure that the units are in the SI system using openmdao convert unit_function
    root_chord_original_value = var_outputs_compute_root_chord.get("data:geometry:wing:root:chord")[
        0
    ]
    root_chord_original_unit = var_outputs_compute_root_chord.get("data:geometry:wing:root:chord")[
        1
    ]
    root_chord = convert_units(root_chord_original_value, root_chord_original_unit, "m")

    # We now compute the tip chord

    var_inputs_compute_tip_chord = copy.copy(var_inputs_compute_root_chord)

    compute_wing_tip_chord = generate_block_analysis(
        ComputeWingL1AndL4(), var_inputs_compute_tip_chord, "", False
    )

    var_dict_compute_root_chord = copy.copy(var_dict_compute_root_chord)

    var_outputs_compute_tip_chord = compute_wing_tip_chord(var_dict_compute_root_chord)

    # We ensure that the units are in the SI system using openmdao convert unit_function
    tip_chord_original_value = var_outputs_compute_tip_chord.get("data:geometry:wing:tip:chord")[0]
    tip_chord_original_unit = var_outputs_compute_tip_chord.get("data:geometry:wing:tip:chord")[1]
    tip_chord = convert_units(tip_chord_original_value, tip_chord_original_unit, "m")

    # We can now move on to the computation of the mfw for that wing area and then return the
    # difference to solve for the right wing_area

    var_inputs_compute_mfw = [
        "data:propulsion:fuel_type",
        "data:geometry:wing:root:chord",
        "data:geometry:wing:tip:chord",
        "data:geometry:wing:root:y",
        "data:geometry:wing:tip:y",
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
        "data:geometry:wing:span",
        "data:geometry:landing_gear:type",
        "data:geometry:landing_gear:y",
        "settings:geometry:fuel_tanks:depth",
    ]

    compute_mfw = generate_block_analysis(ComputeMFWAdvanced(), var_inputs_compute_mfw, "", False)

    var_dict_compute_root_chord = {
        "data:propulsion:fuel_type": (fuel_type, None),
        "data:geometry:wing:root:chord": (root_chord, "m"),
        "data:geometry:wing:tip:chord": (tip_chord, "m"),
        "data:geometry:wing:root:y": (root_y, "m"),
        "data:geometry:wing:tip:y": (tip_y, "m"),
        "data:geometry:wing:root:thickness_ratio": (root_tc, None),
        "data:geometry:wing:tip:thickness_ratio": (tip_tc, None),
        "data:geometry:flap:chord_ratio": (flap_chord_ratio, None),
        "data:geometry:wing:aileron:chord_ratio": (aileron_chord_ratio, None),
        "data:geometry:propulsion:tank:y_ratio_tank_beginning": (y_ratio_tank_beginning, None),
        "data:geometry:propulsion:tank:y_ratio_tank_end": (y_ratio_tank_end, None),
        "data:geometry:propulsion:engine:layout": (engine_config, None),
        "data:geometry:propulsion:engine:y_ratio": (y_ratio_tank, None),
        "data:geometry:propulsion:tank:LE_chord_percentage": (le_chord_percentage, None),
        "data:geometry:propulsion:tank:TE_chord_percentage": (te_chord_percentage, None),
        "data:geometry:propulsion:nacelle:width": (nacelle_width, "m"),
        "data:geometry:wing:span": (wing_span, "m"),
        "data:geometry:landing_gear:type": (lg_type, None),
        "data:geometry:landing_gear:y": (y_lg, "m"),
        "settings:geometry:fuel_tanks:depth": (k, None),
    }

    var_outputs_compute_mfw = compute_mfw(var_dict_compute_root_chord)

    mfw_original_val = var_outputs_compute_mfw.get("data:weight:aircraft:MFW")[0]
    mfw_original_unit = var_outputs_compute_mfw.get("data:weight:aircraft:MFW")[1]
    mfw = convert_units(mfw_original_val, mfw_original_unit, "kg")

    return mfw - fuel_mission

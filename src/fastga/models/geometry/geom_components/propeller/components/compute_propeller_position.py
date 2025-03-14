"""
Python module for propeller position calculation wrt the wing, part of the propeller component.
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

import warnings
import numpy as np
import openmdao.api as om
import fastoad.api as oad

from ..constants import SERVICE_PROPELLER_POSITION, SUBMODEL_PROPELLER_POSITION_LEGACY


@oad.RegisterSubmodel(SERVICE_PROPELLER_POSITION, SUBMODEL_PROPELLER_POSITION_LEGACY)
class ComputePropellerPosition(om.ExplicitComponent):
    """Propeller position with respect to the leading edge estimation."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input(
            "data:geometry:propulsion:engine:y_ratio",
            shape_by_conn=True,
        )
        self.add_input(
            "data:geometry:propulsion:nacelle:x",
            shape_by_conn=True,
            copy_shape="data:geometry:propulsion:engine:y_ratio",
            units="m",
        )
        self.add_input("data:geometry:propulsion:nacelle:length", val=np.nan, units="m")

        self.add_output(
            "data:geometry:propulsion:nacelle:from_LE",
            shape_by_conn=True,
            copy_shape="data:geometry:propulsion:engine:y_ratio",
            units="m",
        )

        self.declare_partials("*", "data:geometry:propulsion:engine:layout", method="fd")

        self.declare_partials("*", "*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        prop_layout = inputs["data:geometry:propulsion:engine:layout"]
        span = inputs["data:geometry:wing:span"]
        y_ratio = np.array(inputs["data:geometry:propulsion:engine:y_ratio"])
        y2_wing = inputs["data:geometry:wing:root:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        nacelle_length = inputs["data:geometry:propulsion:nacelle:length"]
        nacelle_x = np.array(inputs["data:geometry:propulsion:nacelle:x"])

        if prop_layout == 1.0:
            y_nacelle_array = y_ratio * span / 2.0

            tapered_mask = y_nacelle_array > y2_wing

            chord_array = np.full_like(y_nacelle_array, l2_wing)

            chord_array[tapered_mask] = l2_wing + (l4_wing - l2_wing) / (y4_wing - y2_wing) * (
                y_nacelle_array[tapered_mask] - y2_wing
            )

            x_from_le_array = np.maximum(
                np.full_like(chord_array, nacelle_length) - chord_array, 0.0
            )

        elif prop_layout == 2.0:
            x_from_le_array = fa_length - 0.25 * l0_wing - (nacelle_x[0] - nacelle_length)
        elif prop_layout == 3.0:
            x_from_le_array = fa_length - 0.25 * l0_wing
        else:
            x_from_le_array = fa_length - 0.25 * l0_wing
            warnings.warn(
                "Propulsion layout {} not implemented in model, replaced by layout 3!".format(
                    prop_layout
                )
            )

        outputs["data:geometry:propulsion:nacelle:from_LE"] = x_from_le_array

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        prop_layout = inputs["data:geometry:propulsion:engine:layout"]
        span = inputs["data:geometry:wing:span"]
        y_ratio = np.array(inputs["data:geometry:propulsion:engine:y_ratio"])
        y2_wing = inputs["data:geometry:wing:root:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        nacelle_length = inputs["data:geometry:propulsion:nacelle:length"]

        if prop_layout == 1.0:
            y_nacelle_array = y_ratio * span / 2.0

            tapered_mask = y_nacelle_array > y2_wing

            chord_array = np.full_like(y_nacelle_array, l2_wing)
            y_nacelle_masked = y_nacelle_array[tapered_mask]
            chord_array[tapered_mask] = l2_wing + (l4_wing - l2_wing) / (y4_wing - y2_wing) * (
                y_nacelle_masked - y2_wing
            )

            nacelle_chord_mask = chord_array > nacelle_length

            derivative_wrt_nacelle_length = np.ones_like(y_ratio)

            derivative_wrt_l4_wing = _set_value(
                np.zeros_like(y_ratio),
                tapered_mask,
                -(y_nacelle_masked - y2_wing) / (y4_wing - y2_wing),
            )

            derivative_wrt_y4_wing = _set_value(
                np.zeros_like(y_ratio),
                tapered_mask,
                (-(l4_wing - l2_wing) * (y_nacelle_masked - y2_wing) / (y4_wing - y2_wing) ** 2.0),
            )

            derivative_wrt_l2_wing = _set_value(
                -np.ones_like(y_ratio),
                tapered_mask,
                -(1.0 - (y_nacelle_masked - y2_wing) / (y4_wing - y2_wing)),
            )

            derivative_wrt_y2_wing = _set_value(
                np.zeros_like(y_ratio),
                tapered_mask,
                (
                    -(l4_wing - l2_wing)
                    * (-(y4_wing - y2_wing) + (y_nacelle_masked - y2_wing))
                    / (y4_wing - y2_wing) ** 2.0
                ),
            )

            derivative_wrt_span = _set_value(
                np.zeros_like(y_ratio),
                tapered_mask,
                (-(l4_wing - l2_wing) / (y4_wing - y2_wing) * y_ratio[tapered_mask] / 2.0),
            )

            derivative_wrt_y_ratio = _set_value(
                np.zeros((len(y_nacelle_array), len(y_nacelle_array))),
                tapered_mask,
                (-(l4_wing - l2_wing) / (y4_wing - y2_wing) * span / 2.0),
            )

            partials[
                "data:geometry:propulsion:nacelle:from_LE",
                "data:geometry:propulsion:nacelle:length",
            ] = _set_value(derivative_wrt_nacelle_length, nacelle_chord_mask, 0.0)

            partials["data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:tip:chord"] = (
                _set_value(derivative_wrt_l4_wing, nacelle_chord_mask, 0.0)
            )
            partials["data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:tip:y"] = (
                _set_value(derivative_wrt_y4_wing, nacelle_chord_mask, 0.0)
            )
            partials[
                "data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:root:chord"
            ] = _set_value(derivative_wrt_l2_wing, nacelle_chord_mask, 0.0)

            partials["data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:root:y"] = (
                _set_value(derivative_wrt_y2_wing, nacelle_chord_mask, 0.0)
            )

            partials["data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:span"] = (
                _set_value(derivative_wrt_span, nacelle_chord_mask, 0.0)
            )

            partials[
                "data:geometry:propulsion:nacelle:from_LE",
                "data:geometry:propulsion:engine:y_ratio",
            ] = _set_value(derivative_wrt_y_ratio, nacelle_chord_mask, 0.0)

        elif prop_layout == 2.0:
            partials[
                "data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:MAC:at25percent:x"
            ] = 1.0
            partials[
                "data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:MAC:length"
            ] = -0.25
            partials[
                "data:geometry:propulsion:nacelle:from_LE",
                "data:geometry:propulsion:nacelle:length",
            ] = 1.0
            partials[
                "data:geometry:propulsion:nacelle:from_LE", "data:geometry:propulsion:nacelle:x"
            ] = -1.0
            partials["data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:root:y"] = 0.0
            partials[
                "data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:root:chord"
            ] = 0.0
            partials["data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:tip:y"] = 0.0
            partials["data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:tip:chord"] = (
                0.0
            )
            partials["data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:span"] = 0.0
            partials[
                "data:geometry:propulsion:nacelle:from_LE",
                "data:geometry:propulsion:engine:y_ratio",
            ] = 0.0

        else:
            partials[
                "data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:MAC:at25percent:x"
            ] = 1.0
            partials[
                "data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:MAC:length"
            ] = -0.25
            partials[
                "data:geometry:propulsion:nacelle:from_LE",
                "data:geometry:propulsion:nacelle:length",
            ] = 0.0
            partials[
                "data:geometry:propulsion:nacelle:from_LE", "data:geometry:propulsion:nacelle:x"
            ] = 0.0
            partials["data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:root:y"] = 0.0
            partials[
                "data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:root:chord"
            ] = 0.0
            partials["data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:tip:y"] = 0.0
            partials["data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:tip:chord"] = (
                0.0
            )
            partials["data:geometry:propulsion:nacelle:from_LE", "data:geometry:wing:span"] = 0.0
            partials[
                "data:geometry:propulsion:nacelle:from_LE",
                "data:geometry:propulsion:engine:y_ratio",
            ] = 0.0


def _set_value(array, mask, value):
    if array.ndim == 2:
        array[mask, mask] = value
    else:
        array[mask] = value
    return array

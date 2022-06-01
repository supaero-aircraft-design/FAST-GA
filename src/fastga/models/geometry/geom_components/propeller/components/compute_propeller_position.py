"""Estimation of propeller position wrt to the wing."""
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

import warnings

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_PROPELLER_POSITION


@oad.RegisterSubmodel(
    SUBMODEL_PROPELLER_POSITION, "fastga.submodel.geometry.propeller.position.legacy"
)
class ComputePropellerPosition(om.ExplicitComponent):
    """Propeller position with respect to the leading edge estimation."""

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

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prop_layout = inputs["data:geometry:propulsion:engine:layout"]
        span = inputs["data:geometry:wing:span"]
        y_ratio = np.array(inputs["data:geometry:propulsion:engine:y_ratio"])
        y2_wing = float(inputs["data:geometry:wing:root:y"])
        l2_wing = float(inputs["data:geometry:wing:root:chord"])
        l0_wing = float(inputs["data:geometry:wing:MAC:length"])
        fa_length = float(inputs["data:geometry:wing:MAC:at25percent:x"])
        y4_wing = float(inputs["data:geometry:wing:tip:y"])
        l4_wing = float(inputs["data:geometry:wing:tip:chord"])
        nacelle_length = float(inputs["data:geometry:propulsion:nacelle:length"])
        nacelle_x = np.array(inputs["data:geometry:propulsion:nacelle:x"])

        if prop_layout == 1.0:
            y_nacelle_array = y_ratio * span / 2

            x_from_le_array = np.copy(y_nacelle_array)

            for idx, y_nacelle in enumerate(y_nacelle_array):
                if y_nacelle > y2_wing:  # Nacelle in the tapered part of the wing
                    chord = l2_wing + (l4_wing - l2_wing) / (y4_wing - y2_wing) * (
                        y_nacelle - y2_wing
                    )
                else:  # Nacelle in the straight part of the wing
                    chord = l2_wing

                x_from_le_array[idx] = max(nacelle_length - chord, 0.0)

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

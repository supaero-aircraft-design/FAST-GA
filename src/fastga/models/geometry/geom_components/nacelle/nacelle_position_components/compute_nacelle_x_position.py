"""
Python module for nacelle X - position calculation, part of the nacelle position.
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


class ComputeNacelleXPosition(om.ExplicitComponent):
    """
    Estimates x position of the nacelle.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:length", val=np.nan, units="m")
        self.add_input(
            "data:geometry:propulsion:nacelle:y",
            val=np.nan,
            units="m",
            shape_by_conn=True,
        )

        self.add_output(
            "data:geometry:propulsion:nacelle:x",
            units="m",
            shape_by_conn=True,
            copy_shape="data:geometry:propulsion:nacelle:y",
        )

        self.declare_partials(
            of="*",
            wrt=[
                "data:geometry:wing:tip:y",
                "data:geometry:wing:tip:leading_edge:x:local",
                "data:geometry:wing:root:y",
                "data:geometry:wing:MAC:leading_edge:x:local",
                "data:geometry:wing:MAC:at25percent:x",
                "data:geometry:wing:MAC:length",
                "data:geometry:fuselage:length",
                "data:geometry:fuselage:rear_length",
                "data:geometry:propulsion:nacelle:length",
                "data:geometry:propulsion:nacelle:y",
            ],
            method="exact",
        )

        self.declare_partials(of="*", wrt="data:geometry:propulsion:engine:layout", method="fd")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        nac_length = inputs["data:geometry:propulsion:nacelle:length"]
        prop_layout = inputs["data:geometry:propulsion:engine:layout"]
        fus_length = inputs["data:geometry:fuselage:length"]
        rear_length = inputs["data:geometry:fuselage:rear_length"]
        y2_wing = float(inputs["data:geometry:wing:root:y"])
        x0_wing = float(inputs["data:geometry:wing:MAC:leading_edge:x:local"])
        l0_wing = float(inputs["data:geometry:wing:MAC:length"])
        fa_length = float(inputs["data:geometry:wing:MAC:at25percent:x"])
        x4_wing = float(inputs["data:geometry:wing:tip:leading_edge:x:local"])
        y4_wing = float(inputs["data:geometry:wing:tip:y"])

        if prop_layout == 1.0:
            y_nacelle_array = inputs["data:geometry:propulsion:nacelle:y"]
            x_nacelle_array = np.copy(y_nacelle_array)

            for idx, y_nacelle in enumerate(y_nacelle_array):
                if y_nacelle > y2_wing:  # Nacelle in the tapered part of the wing
                    delta_x_nacelle = x4_wing * (y_nacelle - y2_wing) / (y4_wing - y2_wing)
                else:  # Nacelle in the straight part of the wing
                    delta_x_nacelle = 0
                x_nacelle_array[idx] = fa_length - x0_wing - 0.25 * l0_wing + delta_x_nacelle

        elif prop_layout == 2.0:
            x_nacelle_array = fus_length - 0.1 * rear_length
        elif prop_layout == 3.0:
            x_nacelle_array = float(nac_length)
        else:
            x_nacelle_array = float(nac_length)

            warnings.warn(
                "Propulsion layout {} not implemented in model, replaced by layout 3!".format(
                    prop_layout
                )
            )

        outputs["data:geometry:propulsion:nacelle:x"] = x_nacelle_array

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        prop_layout = inputs["data:geometry:propulsion:engine:layout"]
        y2_wing = float(inputs["data:geometry:wing:root:y"])
        x4_wing = float(inputs["data:geometry:wing:tip:leading_edge:x:local"])
        y4_wing = float(inputs["data:geometry:wing:tip:y"])

        if prop_layout == 1.0:
            y_nacelle_array = inputs["data:geometry:propulsion:nacelle:y"]
            d_x_nacelle_d_y2_wing = np.copy(y_nacelle_array)
            d_x_nacelle_d_x4_wing = np.copy(y_nacelle_array)
            d_x_nacelle_d_y4_wing = np.copy(y_nacelle_array)
            d_x_nacelle_d_y_nacelle = np.zeros((len(y_nacelle_array), len(y_nacelle_array)))

            for idx, y_nacelle in enumerate(y_nacelle_array):
                if y_nacelle > y2_wing:  # Nacelle in the tapered part of the wing
                    d_x_nacelle_d_y2_wing[idx] = (
                        x4_wing
                        * (-(y4_wing - y2_wing) + (y_nacelle - y2_wing))
                        / (y4_wing - y2_wing) ** 2
                    )
                    d_x_nacelle_d_x4_wing[idx] = (y_nacelle - y2_wing) / (y4_wing - y2_wing)
                    d_x_nacelle_d_y4_wing[idx] = (
                        -x4_wing * (y_nacelle - y2_wing) / (y4_wing - y2_wing) ** 2
                    )
                    d_x_nacelle_d_y_nacelle[idx, idx] = x4_wing / (y4_wing - y2_wing)
                else:  # Nacelle in the straight part of the wing
                    d_x_nacelle_d_y2_wing[idx] = 0.0
                    d_x_nacelle_d_x4_wing[idx] = 0.0
                    d_x_nacelle_d_y4_wing[idx] = 0.0

            partials[
                "data:geometry:propulsion:nacelle:x", "data:geometry:propulsion:nacelle:length"
            ] = 0.0
            partials["data:geometry:propulsion:nacelle:x", "data:geometry:fuselage:length"] = 0.0
            partials["data:geometry:propulsion:nacelle:x", "data:geometry:fuselage:rear_length"] = (
                0.0
            )
            partials["data:geometry:propulsion:nacelle:x", "data:geometry:wing:root:y"] = (
                d_x_nacelle_d_y2_wing
            )
            partials[
                "data:geometry:propulsion:nacelle:x", "data:geometry:wing:MAC:leading_edge:x:local"
            ] = -1.0
            partials["data:geometry:propulsion:nacelle:x", "data:geometry:wing:MAC:length"] = -0.25
            partials[
                "data:geometry:propulsion:nacelle:x", "data:geometry:wing:MAC:at25percent:x"
            ] = 1.0
            partials[
                "data:geometry:propulsion:nacelle:x", "data:geometry:wing:tip:leading_edge:x:local"
            ] = d_x_nacelle_d_x4_wing
            partials["data:geometry:propulsion:nacelle:x", "data:geometry:wing:tip:y"] = (
                d_x_nacelle_d_y4_wing
            )
            partials["data:geometry:propulsion:nacelle:x", "data:geometry:propulsion:nacelle:y"] = (
                d_x_nacelle_d_y_nacelle
            )

        elif prop_layout == 2.0:
            partials[
                "data:geometry:propulsion:nacelle:x", "data:geometry:propulsion:nacelle:length"
            ] = 0.0
            partials["data:geometry:propulsion:nacelle:x", "data:geometry:fuselage:length"] = 1.0
            partials[
                "data:geometry:propulsion:nacelle:x", "data:geometry:fuselage:rear_length"
            ] = -0.1
            partials["data:geometry:propulsion:nacelle:x", "data:geometry:wing:root:y"] = 0.0
            partials[
                "data:geometry:propulsion:nacelle:x", "data:geometry:wing:MAC:leading_edge:x:local"
            ] = 0.0
            partials["data:geometry:propulsion:nacelle:x", "data:geometry:wing:MAC:length"] = 0.0
            partials[
                "data:geometry:propulsion:nacelle:x", "data:geometry:wing:MAC:at25percent:x"
            ] = 0.0
            partials[
                "data:geometry:propulsion:nacelle:x", "data:geometry:wing:tip:leading_edge:x:local"
            ] = 0.0
            partials["data:geometry:propulsion:nacelle:x", "data:geometry:wing:tip:y"] = 0.0
            partials["data:geometry:propulsion:nacelle:x", "data:geometry:propulsion:nacelle:y"] = (
                0.0
            )

        else:
            partials[
                "data:geometry:propulsion:nacelle:x", "data:geometry:propulsion:nacelle:length"
            ] = 1.0
            partials["data:geometry:propulsion:nacelle:x", "data:geometry:fuselage:length"] = 0.0
            partials["data:geometry:propulsion:nacelle:x", "data:geometry:fuselage:rear_length"] = (
                0.0
            )
            partials["data:geometry:propulsion:nacelle:x", "data:geometry:wing:root:y"] = 0.0
            partials[
                "data:geometry:propulsion:nacelle:x", "data:geometry:wing:MAC:leading_edge:x:local"
            ] = 0.0
            partials["data:geometry:propulsion:nacelle:x", "data:geometry:wing:MAC:length"] = 0.0
            partials[
                "data:geometry:propulsion:nacelle:x", "data:geometry:wing:MAC:at25percent:x"
            ] = 0.0
            partials[
                "data:geometry:propulsion:nacelle:x", "data:geometry:wing:tip:leading_edge:x:local"
            ] = 0.0
            partials["data:geometry:propulsion:nacelle:x", "data:geometry:wing:tip:y"] = 0.0
            partials["data:geometry:propulsion:nacelle:x", "data:geometry:propulsion:nacelle:y"] = (
                0.0
            )

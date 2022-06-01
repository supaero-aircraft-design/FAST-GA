"""Estimation of nacelle and pylon geometry."""
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
import warnings
import openmdao.api as om

import fastoad.api as oad

from ...constants import SUBMODEL_NACELLE_POSITION


@oad.RegisterSubmodel(SUBMODEL_NACELLE_POSITION, "fastga.submodel.geometry.nacelle.position.legacy")
class ComputeNacellePosition(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Nacelle and pylon geometry estimation."""

    def setup(self):

        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input(
            "data:geometry:propulsion:engine:y_ratio",
            shape_by_conn=True,
        )
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:width", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:length", val=np.nan, units="m")

        self.add_output(
            "data:geometry:propulsion:nacelle:y",
            units="m",
            shape_by_conn=True,
            copy_shape="data:geometry:propulsion:engine:y_ratio",
        )
        self.add_output(
            "data:geometry:propulsion:nacelle:x",
            units="m",
            shape_by_conn=True,
            copy_shape="data:geometry:propulsion:engine:y_ratio",
        )

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        nac_width = inputs["data:geometry:propulsion:nacelle:width"]
        nac_length = inputs["data:geometry:propulsion:nacelle:length"]
        prop_layout = inputs["data:geometry:propulsion:engine:layout"]
        span = inputs["data:geometry:wing:span"]
        y_ratio = np.array(inputs["data:geometry:propulsion:engine:y_ratio"])
        b_f = inputs["data:geometry:fuselage:maximum_width"]
        fus_length = inputs["data:geometry:fuselage:length"]
        rear_length = inputs["data:geometry:fuselage:rear_length"]
        y2_wing = float(inputs["data:geometry:wing:root:y"])
        x0_wing = float(inputs["data:geometry:wing:MAC:leading_edge:x:local"])
        l0_wing = float(inputs["data:geometry:wing:MAC:length"])
        fa_length = float(inputs["data:geometry:wing:MAC:at25percent:x"])
        x4_wing = float(inputs["data:geometry:wing:tip:leading_edge:x:local"])
        y4_wing = float(inputs["data:geometry:wing:tip:y"])

        if prop_layout == 1.0:
            y_nacelle_array = y_ratio * span / 2
            x_nacelle_array = np.copy(y_nacelle_array)

            for idx, y_nacelle in enumerate(y_nacelle_array):
                if y_nacelle > y2_wing:  # Nacelle in the tapered part of the wing
                    delta_x_nacelle = x4_wing * (y_nacelle - y2_wing) / (y4_wing - y2_wing)
                else:  # Nacelle in the straight part of the wing
                    delta_x_nacelle = 0
                x_nacelle_array[idx] = fa_length - x0_wing - 0.25 * l0_wing + delta_x_nacelle

        elif prop_layout == 2.0:
            y_nacelle_array = b_f / 2 + 0.8 * nac_width
            x_nacelle_array = fus_length - 0.1 * rear_length
        elif prop_layout == 3.0:
            y_nacelle_array = 0.0
            x_nacelle_array = float(nac_length)
        else:
            y_nacelle_array = 0.0
            x_nacelle_array = float(nac_length)

            warnings.warn(
                "Propulsion layout {} not implemented in model, replaced by layout 3!".format(
                    prop_layout
                )
            )

        outputs["data:geometry:propulsion:nacelle:y"] = y_nacelle_array
        outputs["data:geometry:propulsion:nacelle:x"] = x_nacelle_array

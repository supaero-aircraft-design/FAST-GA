"""Estimation of engine(s) center of gravity."""
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

from openmdao.core.explicitcomponent import ExplicitComponent


class ComputeEngineCG(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Engine(s) center of gravity estimation"""

    def setup(self):

        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:length", val=np.nan, units="m")
        self.add_input(
            "data:geometry:propulsion:nacelle:y", val=np.nan, shape_by_conn=True, units="m"
        )
        self.add_input(
            "data:geometry:propulsion:nacelle:x",
            val=np.nan,
            shape_by_conn=True,
            copy_shape="data:geometry:propulsion:nacelle:y",
            units="m",
        )
        self.add_input("data:geometry:propeller:depth", val=np.nan, units="m")

        self.add_output("data:weight:propulsion:engine:CG:x", units="m")

        self.declare_partials(
            "data:weight:propulsion:engine:CG:x",
            [
                "data:geometry:wing:root:y",
                "data:geometry:wing:root:chord",
                "data:geometry:wing:tip:y",
                "data:geometry:wing:tip:chord",
                "data:geometry:propulsion:nacelle:length",
                "data:geometry:propulsion:nacelle:y",
                "data:geometry:propeller:depth",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prop_layout = inputs["data:geometry:propulsion:engine:layout"]
        engine_count_pre_wing = inputs["data:geometry:propulsion:engine:count"] / 2.0
        y2_wing = inputs["data:geometry:wing:root:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        nacelle_length = inputs["data:geometry:propulsion:nacelle:length"]
        y_nacelle_array = inputs["data:geometry:propulsion:nacelle:y"]
        x_nacelle_array = inputs["data:geometry:propulsion:nacelle:x"]
        prop_depth = inputs["data:geometry:propeller:depth"]

        x_cg_in_nacelle = 0.6 * nacelle_length
        # From the beginning of the nacelle wrt to the nose, the CG is at x_cg_in_nacelle

        if prop_layout == 1.0:

            x_cg_b1 = 0

            for y_nacelle, x_nacelle in zip(y_nacelle_array, x_nacelle_array):
                if y_nacelle > y2_wing:  # Nacelle in the tapered part of the wing
                    l_wing_nac = l4_wing + (l2_wing - l4_wing) * (y4_wing - y_nacelle) / (
                        y4_wing - y2_wing
                    )
                    delta_x_nacelle = 0.05 * l_wing_nac
                    x_nacelle_cg = x_nacelle - delta_x_nacelle - (nacelle_length - x_cg_in_nacelle)
                else:  # Nacelle in the straight part of the wing
                    l_wing_nac = l2_wing
                    delta_x_nacelle = 0.05 * l_wing_nac
                    x_nacelle_cg = x_nacelle - delta_x_nacelle - (nacelle_length - x_cg_in_nacelle)
                x_cg_b1 += x_nacelle_cg / engine_count_pre_wing
        elif prop_layout == 2.0:
            x_cg_b1 = x_nacelle_array - (nacelle_length - x_cg_in_nacelle)
        elif prop_layout == 3.0:
            x_cg_b1 = x_cg_in_nacelle + prop_depth
        else:
            x_cg_b1 = x_cg_in_nacelle + prop_depth
            warnings.warn(
                "Propulsion layout {} not implemented in model, replaced by layout 3!".format(
                    prop_layout
                )
            )

        outputs["data:weight:propulsion:engine:CG:x"] = x_cg_b1

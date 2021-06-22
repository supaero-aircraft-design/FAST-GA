"""
    Estimation of nacelle and pylon geometry
"""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader

from fastga.models.propulsion.fuel_propulsion.base import FuelEngineSet

from fastga.models.aerodynamics.constants import ENGINE_COUNT


class ComputeNacelleGeometry(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Nacelle and pylon geometry estimation """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:y_ratio", shape=ENGINE_COUNT, val=np.nan)
        self.add_input("data:geometry:propulsion:count", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
        self.add_input("data:geometry:propeller:diameter", val=np.nan, units="m")

        self.add_output("data:geometry:propulsion:nacelle:length", units="m")
        self.add_output("data:geometry:propulsion:nacelle:height", units="m")
        self.add_output("data:geometry:propulsion:nacelle:width", units="m")
        self.add_output("data:geometry:propulsion:nacelle:wet_area", units="m**2")
        self.add_output("data:geometry:propulsion:propeller:depth", units="m")
        self.add_output("data:geometry:propulsion:propeller:diameter", units="m")
        self.add_output("data:geometry:propulsion:nacelle:y", shape=ENGINE_COUNT, units="m")
        self.add_output("data:geometry:propulsion:nacelle:x", shape=ENGINE_COUNT, units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propulsion_model = FuelEngineSet(self._engine_wrapper.get_model(inputs), 1.0)
        prop_layout = inputs["data:geometry:propulsion:layout"]
        prop_count = inputs["data:geometry:propulsion:count"]
        span = inputs["data:geometry:wing:span"]
        y_ratio = np.array(inputs["data:geometry:propulsion:y_ratio"])
        b_f = inputs["data:geometry:fuselage:maximum_width"]
        fus_length = inputs["data:geometry:fuselage:length"]
        rear_length = inputs["data:geometry:fuselage:rear_length"]
        y2_wing = float(inputs["data:geometry:wing:root:y"])
        x0_wing = float(inputs["data:geometry:wing:MAC:leading_edge:x:local"])
        l0_wing = float(inputs["data:geometry:wing:MAC:length"])
        fa_length = float(inputs["data:geometry:wing:MAC:at25percent:x"])
        x4_wing = float(inputs["data:geometry:wing:tip:leading_edge:x:local"])
        y4_wing = float(inputs["data:geometry:wing:tip:y"])

        nac_height, nac_width, nac_length, nac_wet_area = propulsion_model.compute_dimensions()

        if prop_layout == 1.0:
            y_nacelle_array = y_ratio * span / 2
            unused_index = np.where(y_nacelle_array < 0.0)
            if ENGINE_COUNT - len(unused_index[0]) != int(prop_count / 2.0):
                warnings.warn(
                    "Engine count and engine position do not match, change value in the xml"
                )
            for i in unused_index:
                y_nacelle_array[i] = -1.0

            used_index = np.where(y_nacelle_array >= 0.0)[0]
            x_nacelle_array = np.copy(y_nacelle_array)

            for index in used_index:
                y_nacelle = y_nacelle_array[index]
                if y_nacelle > y2_wing:  # Nacelle in the tapered part of the wing
                    delta_x_nacelle = x4_wing * (y_nacelle - y2_wing) / (y4_wing - y2_wing)
                else:  # Nacelle in the straight part of the wing
                    delta_x_nacelle = 0
                x_nacelle_array[index] = fa_length - x0_wing - 0.25 * l0_wing + delta_x_nacelle

        elif prop_layout == 2.0:
            y_nacelle = b_f / 2 + 0.8 * nac_width
            y_nacelle_array = np.concatenate(
                (np.array([y_nacelle]), np.full(ENGINE_COUNT - 1, -1.0))
            )
            x_nacelle = fus_length - 0.1 * rear_length
            x_nacelle_array = np.concatenate(
                (np.array([x_nacelle]), np.full(ENGINE_COUNT - 1, -1.0))
            )
        elif prop_layout == 3.0:
            y_nacelle = 0.0
            y_nacelle_array = np.concatenate(
                (np.array([y_nacelle]), np.full(ENGINE_COUNT - 1, -1.0))
            )
            x_nacelle = float(nac_length)
            x_nacelle_array = np.concatenate(
                (np.array([x_nacelle]), np.full(ENGINE_COUNT - 1, -1.0))
            )
        else:
            y_nacelle = 0.0
            y_nacelle_array = np.concatenate(
                (np.array([y_nacelle]), np.full(ENGINE_COUNT - 1, -1.0))
            )
            x_nacelle = float(nac_length)
            x_nacelle_array = np.concatenate(
                (np.array([x_nacelle]), np.full(ENGINE_COUNT - 1, -1.0))
            )
            warnings.warn(
                "Propulsion layout {} not implemented in model, replaced by layout 3!".format(
                    prop_layout
                )
            )

        outputs["data:geometry:propulsion:nacelle:length"] = nac_length
        outputs["data:geometry:propulsion:nacelle:height"] = nac_height
        outputs["data:geometry:propulsion:nacelle:width"] = nac_width
        outputs["data:geometry:propulsion:nacelle:wet_area"] = nac_wet_area
        outputs["data:geometry:propulsion:nacelle:y"] = y_nacelle_array
        outputs["data:geometry:propulsion:nacelle:x"] = x_nacelle_array

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
        self.add_input("data:geometry:propulsion:y_ratio", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:propeller:diameter", val=np.nan, units="m")

        self.add_output("data:geometry:propulsion:nacelle:length", units="m")
        self.add_output("data:geometry:propulsion:nacelle:height", units="m")
        self.add_output("data:geometry:propulsion:nacelle:width", units="m")
        self.add_output("data:geometry:propulsion:nacelle:wet_area", units="m**2")
        self.add_output("data:geometry:propulsion:propeller:depth", units="m")
        self.add_output("data:geometry:propulsion:propeller:diameter", units="m")
        self.add_output("data:geometry:landing_gear:height", units="m")
        self.add_output("data:geometry:propulsion:nacelle:y", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propulsion_model = FuelEngineSet(self._engine_wrapper.get_model(inputs), 1.0)
        prop_layout = inputs["data:geometry:propulsion:layout"]
        span = inputs["data:geometry:wing:span"]
        y_ratio = inputs["data:geometry:propulsion:y_ratio"]
        b_f = inputs["data:geometry:fuselage:maximum_width"]

        nac_height, nac_width, nac_length, nac_wet_area = propulsion_model.compute_dimensions()

        if prop_layout == 1.0:
            y_nacelle = y_ratio * span / 2
        elif prop_layout == 2.0:
            y_nacelle = b_f / 2 + 0.8 * nac_width
        elif prop_layout == 3.0:
            y_nacelle = 0.0
        else:
            y_nacelle = y_ratio * span / 2
            warnings.warn(
                "Propulsion layout {} not implemented in model, replaced by layout 1!".format(
                    prop_layout
                )
            )

        lg_height = 0.41 * inputs["data:geometry:propeller:diameter"]

        outputs["data:geometry:propulsion:nacelle:length"] = nac_length
        outputs["data:geometry:propulsion:nacelle:height"] = nac_height
        outputs["data:geometry:propulsion:nacelle:width"] = nac_width
        outputs["data:geometry:propulsion:nacelle:wet_area"] = nac_wet_area
        outputs["data:geometry:landing_gear:height"] = lg_height
        outputs["data:geometry:propulsion:nacelle:y"] = y_nacelle

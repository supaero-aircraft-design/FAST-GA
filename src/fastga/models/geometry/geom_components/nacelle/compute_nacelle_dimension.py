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


import openmdao.api as om

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
import fastoad.api as oad

from fastga.models.propulsion.fuel_propulsion.base import FuelEngineSet

from ...constants import SUBMODEL_NACELLE_DIMENSION


@oad.RegisterSubmodel(
    SUBMODEL_NACELLE_DIMENSION, "fastga.submodel.geometry.nacelle.dimension.legacy"
)
class ComputeNacelleDimension(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Nacelle and pylon geometry estimation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_output("data:geometry:propulsion:nacelle:length", units="m")
        self.add_output("data:geometry:propulsion:nacelle:height", units="m")
        self.add_output("data:geometry:propulsion:nacelle:width", units="m")
        self.add_output("data:geometry:propulsion:nacelle:wet_area", units="m**2")
        self.add_output("data:geometry:propulsion:nacelle:master_cross_section", units="m**2")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propulsion_model = FuelEngineSet(self._engine_wrapper.get_model(inputs), 1.0)

        nac_height, nac_width, nac_length, nac_wet_area = propulsion_model.compute_dimensions()
        master_cross_section = nac_height * nac_width

        outputs["data:geometry:propulsion:nacelle:length"] = nac_length
        outputs["data:geometry:propulsion:nacelle:height"] = nac_height
        outputs["data:geometry:propulsion:nacelle:width"] = nac_width
        outputs["data:geometry:propulsion:nacelle:wet_area"] = nac_wet_area
        outputs["data:geometry:propulsion:nacelle:master_cross_section"] = master_cross_section

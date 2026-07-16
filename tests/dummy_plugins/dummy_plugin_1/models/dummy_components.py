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
import fastoad.api as oad

from fastga.utils.options_checkers import check_propulsion_id

from .constants import DUMMY_SERVICE


@oad.RegisterSubmodel(DUMMY_SERVICE, "dummy_submodel_1")
class DummyComponent1(om.ExplicitComponent):

    def initialize(self):
        # Actually "requires" the propulsion id in the right format
        self.options.declare("propulsion_id", check_valid=check_propulsion_id)

    def setup(self):
        self.add_input(name="dummy_input_1", val=1.0, units="m**2")

        self.add_output(name="dummy_output_1", val=42.0, units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["dummy_output_1"] = 10 + 32.0 * inputs["dummy_input_1"]


@oad.RegisterSubmodel(DUMMY_SERVICE, "dummy_submodel_2")
class DummyComponent2(om.ExplicitComponent):

    def initialize(self):
        # Does not need the option but has it anyway for compatibility
        self.options.declare("propulsion_id", default=None, allow_none=True)

    def setup(self):
        self.add_input(name="dummy_input_2", val=41.46875, units="m**2")

        self.add_output(name="dummy_output_1", val=42.0, units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["dummy_output_1"] = 10 + 32.0 * inputs["dummy_input_1"]

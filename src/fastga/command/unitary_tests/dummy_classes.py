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
import openmdao.api as om

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain


class Disc1(om.ExplicitComponent):
    """An OpenMDAO component to encapsulate Disc1 discipline and test"""

    def setup(self):
        self.add_input(
            "data:geometry:variable_1", val=np.nan, desc=""
        )  # NaN as default for testing connexion check
        self.add_input(
            "data:geometry:variable_2", val=[5, 2], desc="", units="m**2"
        )  # for testing non-None units
        self.add_input("data:geometry:variable_3", val=1.0, desc="")

        self.add_output("data:geometry:variable_4", val=1.0, desc="")
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Evaluates a simple equation
        """
        x1 = inputs["data:geometry:variable_1"][0]
        x21 = inputs["data:geometry:variable_2"][0]
        x22 = inputs["data:geometry:variable_2"][1]
        x3 = inputs["data:geometry:variable_3"][0]

        outputs["data:geometry:variable_4"] = x1 + x21 * x22 - x3


class Disc2(om.Group):
    """An OpenMDAO component to encapsulate Disc1 and an IVC"""

    def setup(self):
        ivc = om.IndepVarComp()
        ivc.add_output("data:geometry:variable_1", val=6.0)
        self.add_subsystem("ivc1", ivc, promotes=["*"])
        self.add_subsystem("disc1", Disc1(), promotes=["*"])


@oad.RegisterOpenMDAOSystem("test.dummy_module.disc3", domain=ModelDomain.OTHER)
class Disc3(om.Group):
    """An OpenMDAO component to encapsulate Disc1, an IVC and option"""

    def initialize(self):
        self.options.declare("ivc_value", types=float, default=6.0)

    def setup(self):
        ivc = om.IndepVarComp()
        ivc.add_output("data:geometry:variable_1", val=self.options["ivc_value"])
        self.add_subsystem("ivc1", ivc, promotes=["*"])
        self.add_subsystem("disc1", Disc1(), promotes=["*"])

#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2026  ONERA & ISAE-SUPAERO
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

from .constants import DUMMY_SERVICE


class DummyGroup(om.Group):

    def initialize(self):
        # Option is not used there strictly speaking, so we allow it to be None
        self.options.declare("propulsion_id", default=None, allow_none=True)

    def setup(self):
        propulsion_option = {"propulsion_id": self.options["propulsion_id"]}

        self.add_subsystem(
            name="dummy_subsystem",
            subsys=oad.RegisterSubmodel.get_submodel(
                service_id=DUMMY_SERVICE,
                options=propulsion_option
            ),
            promotes=["*"]
        )

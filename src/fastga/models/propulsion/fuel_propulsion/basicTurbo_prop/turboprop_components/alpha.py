#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2024  ONERA & ISAE-SUPAERO
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


class AlphaRatio(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("total_pressure_41", units="Pa", shape=n, val=np.nan)
        self.add_input("total_pressure_45", units="Pa", shape=n, val=np.nan)

        self.add_input(
            "data:propulsion:turboprop:design_point:turbine_entry_temperature",
            np.nan,
            units="K",
        )
        self.add_input(
            "total_temperature_45",
            units="degK",
            val=np.full(n, np.nan),
            shape=n,
        )

        self.add_output("data:propulsion:turboprop:design_point:alpha", val=np.full(n, 0.8))
        self.add_output("data:propulsion:turboprop:design_point:alpha_p", val=np.full(n, 0.3))

        self.declare_partials(
            of="data:propulsion:turboprop:design_point:alpha",
            wrt=[
                "data:propulsion:turboprop:design_point:turbine_entry_temperature",
                "total_temperature_45",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:turboprop:design_point:alpha_p",
            wrt=[
                "total_pressure_41",
                "total_pressure_45",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        total_temperature_41 = inputs[
            "data:propulsion:turboprop:design_point:turbine_entry_temperature"
        ]
        total_temperature_45 = inputs["total_temperature_45"]

        total_pressure_45 = inputs["total_pressure_45"]
        total_pressure_41 = inputs["total_pressure_41"]

        outputs["data:propulsion:turboprop:design_point:alpha"] = (
            total_temperature_45 / total_temperature_41
        )
        outputs["data:propulsion:turboprop:design_point:alpha_p"] = (
            total_pressure_45 / total_pressure_41
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        total_temperature_41 = inputs[
            "data:propulsion:turboprop:design_point:turbine_entry_temperature"
        ]
        total_temperature_45 = inputs["total_temperature_45"]

        total_pressure_45 = inputs["total_pressure_45"]
        total_pressure_41 = inputs["total_pressure_41"]

        partials["data:propulsion:turboprop:design_point:alpha", "total_temperature_45"] = np.diag(
            1.0 / total_temperature_41
        )
        partials[
            "data:propulsion:turboprop:design_point:alpha",
            "data:propulsion:turboprop:design_point:turbine_entry_temperature",
        ] = (
            -total_temperature_45 / total_temperature_41 ** 2.0
        )

        partials["data:propulsion:turboprop:design_point:alpha_p", "total_pressure_45"] = np.diag(
            1.0 / total_pressure_41
        )
        partials["data:propulsion:turboprop:design_point:alpha_p", "total_pressure_41"] = -np.diag(
            total_pressure_45 / total_pressure_41 ** 2.0
        )

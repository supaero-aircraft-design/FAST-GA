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


class A45(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("air_mass_flow", units="kg/s", val=np.nan, shape=n)

        self.add_input("gamma_45", shape=n, val=np.nan)

        self.add_input("total_pressure_45", units="Pa", shape=n, val=np.nan)

        self.add_input("fuel_air_ratio", shape=n, val=np.nan)
        self.add_input("compressor_bleed_ratio", shape=n, val=np.nan)
        self.add_input("pressurization_bleed_ratio", shape=n, val=np.nan)

        self.add_input(
            "total_temperature_45",
            np.nan,
            units="K",
        )

        self.add_output("data:propulsion:turboprop:section:45", val=0.00457, units="m**2")

        self.declare_partials(
            of="*",
            wrt=[
                "air_mass_flow",
                "total_pressure_45",
                "total_temperature_45",
                "fuel_air_ratio",
                "compressor_bleed_ratio",
                "pressurization_bleed_ratio",
            ],
            method="exact",
        )
        self.declare_partials(of="*", wrt="gamma_45", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        r_g = 287.0  # Perfect gas constant

        airflow_design = inputs["air_mass_flow"]

        gamma_45 = inputs["gamma_45"]

        total_pressure_45 = inputs["total_pressure_45"]
        total_temperature_45 = inputs["total_temperature_45"]

        fuel_air_ratio = inputs["fuel_air_ratio"]
        compressor_bleed_ratio = inputs["compressor_bleed_ratio"]
        pressurization_bleed_ratio = inputs["pressurization_bleed_ratio"]

        a_45 = (
            airflow_design
            * (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * np.sqrt(total_temperature_45 * r_g)
            / total_pressure_45
            / np.sqrt(gamma_45)
            / (2 / (gamma_45 + 1)) ** ((gamma_45 + 1) / (2 * (gamma_45 - 1)))
        )

        outputs["data:propulsion:turboprop:section:45"] = a_45

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        r_g = 287.0  # Perfect gas constant

        airflow_design = inputs["air_mass_flow"]

        gamma_45 = inputs["gamma_45"]

        total_pressure_45 = inputs["total_pressure_45"]
        total_temperature_45 = inputs["total_temperature_45"]

        fuel_air_ratio = inputs["fuel_air_ratio"]
        compressor_bleed_ratio = inputs["compressor_bleed_ratio"]
        pressurization_bleed_ratio = inputs["pressurization_bleed_ratio"]

        partials["data:propulsion:turboprop:section:45", "air_mass_flow"] = (
            (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * np.sqrt(total_temperature_45 * r_g)
            / total_pressure_45
            / np.sqrt(gamma_45)
            / (2 / (gamma_45 + 1)) ** ((gamma_45 + 1) / (2 * (gamma_45 - 1)))
        )
        partials["data:propulsion:turboprop:section:45", "total_pressure_45"] = -(
            airflow_design
            * (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * np.sqrt(total_temperature_45 * r_g)
            / total_pressure_45 ** 2.0
            / np.sqrt(gamma_45)
            / (2 / (gamma_45 + 1)) ** ((gamma_45 + 1) / (2 * (gamma_45 - 1)))
        )
        partials["data:propulsion:turboprop:section:45", "total_temperature_45",] = (
            0.5
            * airflow_design
            * (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * np.sqrt(r_g / total_temperature_45)
            / total_pressure_45
            / np.sqrt(gamma_45)
            / (2 / (gamma_45 + 1)) ** ((gamma_45 + 1) / (2 * (gamma_45 - 1)))
        )
        partials["data:propulsion:turboprop:section:45", "fuel_air_ratio"] = (
            airflow_design
            * np.sqrt(total_temperature_45 * r_g)
            / total_pressure_45
            / np.sqrt(gamma_45)
            / (2 / (gamma_45 + 1)) ** ((gamma_45 + 1) / (2 * (gamma_45 - 1)))
        )
        partials["data:propulsion:turboprop:section:45", "compressor_bleed_ratio"] = -(
            airflow_design
            * np.sqrt(total_temperature_45 * r_g)
            / total_pressure_45
            / np.sqrt(gamma_45)
            / (2 / (gamma_45 + 1)) ** ((gamma_45 + 1) / (2 * (gamma_45 - 1)))
        )
        partials["data:propulsion:turboprop:section:45", "pressurization_bleed_ratio"] = -(
            airflow_design
            * np.sqrt(total_temperature_45 * r_g)
            / total_pressure_45
            / np.sqrt(gamma_45)
            / (2 / (gamma_45 + 1)) ** ((gamma_45 + 1) / (2 * (gamma_45 - 1)))
        )

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


class A81(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("air_mass_flow", units="kg/s", val=np.nan, shape=n)

        self.add_input("total_pressure_5", units="Pa", shape=n, val=np.nan)

        self.add_input("fuel_air_ratio", shape=n, val=np.nan)
        self.add_input("compressor_bleed_ratio", shape=n, val=np.nan)
        self.add_input("pressurization_bleed_ratio", shape=n, val=np.nan)

        self.add_input(
            "total_temperature_5",
            np.nan,
            units="K",
        )

        self.add_output("data:propulsion:turboprop:section:81", val=0.00457, units="m**2")

        self.declare_partials(
            of="*",
            wrt=[
                "air_mass_flow",
                "total_pressure_5",
                "total_temperature_5",
                "fuel_air_ratio",
                "compressor_bleed_ratio",
                "pressurization_bleed_ratio",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        r_g = 287.0  # Perfect gas constant

        airflow_design = inputs["air_mass_flow"]

        total_pressure_5 = inputs["total_pressure_5"]
        total_temperature_5 = inputs["total_temperature_5"]

        fuel_air_ratio = inputs["fuel_air_ratio"]
        compressor_bleed_ratio = inputs["compressor_bleed_ratio"]
        pressurization_bleed_ratio = inputs["pressurization_bleed_ratio"]

        a_81 = (
            airflow_design
            * (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * np.sqrt(total_temperature_5 * r_g)
            / total_pressure_5
        )

        outputs["data:propulsion:turboprop:section:81"] = a_81

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        r_g = 287.0  # Perfect gas constant

        airflow_design = inputs["air_mass_flow"]

        total_pressure_5 = inputs["total_pressure_5"]
        total_temperature_5 = inputs["total_temperature_5"]

        fuel_air_ratio = inputs["fuel_air_ratio"]
        compressor_bleed_ratio = inputs["compressor_bleed_ratio"]
        pressurization_bleed_ratio = inputs["pressurization_bleed_ratio"]

        partials["data:propulsion:turboprop:section:81", "air_mass_flow"] = (
            (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * np.sqrt(total_temperature_5 * r_g)
            / total_pressure_5
        )
        partials["data:propulsion:turboprop:section:81", "total_pressure_5"] = -(
            airflow_design
            * (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * np.sqrt(total_temperature_5 * r_g)
            / total_pressure_5 ** 2.0
        )
        partials["data:propulsion:turboprop:section:81", "total_temperature_5"] = (
            0.5
            * airflow_design
            * (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * np.sqrt(r_g / total_temperature_5)
            / total_pressure_5
        )
        partials["data:propulsion:turboprop:section:81", "fuel_air_ratio"] = (
            airflow_design * np.sqrt(total_temperature_5 * r_g) / total_pressure_5
        )
        partials["data:propulsion:turboprop:section:81", "compressor_bleed_ratio"] = -(
            airflow_design * np.sqrt(total_temperature_5 * r_g) / total_pressure_5
        )
        partials["data:propulsion:turboprop:section:81", "pressurization_bleed_ratio"] = -(
            airflow_design * np.sqrt(total_temperature_5 * r_g) / total_pressure_5
        )


class A82(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("gamma_5", shape=n, val=np.nan)
        self.add_input("settings:propulsion:turboprop:design_point:mach_exhaust", val=0.4)

        self.add_output("data:propulsion:turboprop:section:82", val=0.00457, units="m**2")

        self.declare_partials(
            of="data:propulsion:turboprop:section:82",
            wrt="settings:propulsion:turboprop:design_point:mach_exhaust",
            method="exact",
        )
        self.declare_partials(of="data:propulsion:turboprop:section:82", wrt="gamma_5", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        gamma_5 = inputs["gamma_5"]
        exhaust_mach = inputs["settings:propulsion:turboprop:design_point:mach_exhaust"]

        a_82 = (
            np.sqrt(gamma_5)
            * exhaust_mach
            * (1 + (gamma_5 - 1) / 2 * exhaust_mach ** 2) ** ((gamma_5 + 1) / (2 * (1 - gamma_5)))
        )

        outputs["data:propulsion:turboprop:section:82"] = a_82

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        gamma_5 = inputs["gamma_5"]
        exhaust_mach = inputs["settings:propulsion:turboprop:design_point:mach_exhaust"]

        partials[
            "data:propulsion:turboprop:section:82",
            "settings:propulsion:turboprop:design_point:mach_exhaust",
        ] = (
            -(2.0 ** (1.0 + (1.0 + gamma_5) / (2.0 * (-1.0 + gamma_5))))
            * np.sqrt(gamma_5)
            * (-1.0 + exhaust_mach ** 2)
            * (2.0 + (-1.0 + gamma_5) * exhaust_mach ** 2.0)
            ** (-3.0 / 2.0 - 1.0 / (-1.0 + gamma_5))
        )


class A8(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        self.add_input("data:propulsion:turboprop:section:81", val=np.nan, units="m**2")
        self.add_input("data:propulsion:turboprop:section:82", val=np.nan, units="m**2")

        self.add_output("data:propulsion:turboprop:section:8", val=0.00457, units="m**2")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["data:propulsion:turboprop:section:8"] = (
            inputs["data:propulsion:turboprop:section:81"]
            / inputs["data:propulsion:turboprop:section:82"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["data:propulsion:turboprop:section:8", "data:propulsion:turboprop:section:81"] = (
            1.0 / inputs["data:propulsion:turboprop:section:82"]
        )
        partials["data:propulsion:turboprop:section:8", "data:propulsion:turboprop:section:82"] = (
            -inputs["data:propulsion:turboprop:section:81"]
            / inputs["data:propulsion:turboprop:section:82"] ** 2.0
        )

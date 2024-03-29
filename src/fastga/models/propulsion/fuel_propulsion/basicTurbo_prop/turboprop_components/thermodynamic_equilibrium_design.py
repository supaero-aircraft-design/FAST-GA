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


class ThermodynamicEquilibriumDesignPoint(om.ImplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("combustion_energy", shape=1, val=0.95 * 43.260e6, units="J/kg")

        self.add_input("cp_2", shape=n, val=np.nan)
        self.add_input("cp_25", shape=n, val=np.nan)
        self.add_input("cp_3", shape=n, val=np.nan)
        self.add_input("cp_4", shape=n, val=np.nan)
        self.add_input("cp_41", shape=n, val=np.nan)
        self.add_input("cp_45", shape=n, val=np.nan)
        self.add_input("cp_5", shape=n, val=np.nan)

        self.add_input("gamma_41", shape=n, val=np.nan)
        self.add_input("gamma_45", shape=n, val=np.nan)
        self.add_input("gamma_5", shape=n, val=np.nan)

        self.add_input("eta_445", shape=1, val=1.0)
        self.add_input("eta_455", shape=1, val=1.0)

        self.add_input("static_pressure_0", units="Pa", shape=n, val=np.nan)

        self.add_input("total_temperature_2", units="K", shape=n, val=np.nan)
        self.add_input("total_temperature_25", units="K", shape=n, val=np.nan)
        self.add_input("total_temperature_3", units="K", shape=n, val=np.nan)

        self.add_input("total_pressure_4", units="Pa", shape=n, val=np.nan)

        self.add_input("fuel_air_ratio", shape=n, val=np.nan)
        self.add_input("compressor_bleed_ratio", shape=n, val=np.nan)
        self.add_input("cooling_bleed_ratio", shape=n, val=np.nan)
        self.add_input("pressurization_bleed_ratio", shape=n, val=np.nan)

        self.add_input(
            "settings:propulsion:turboprop:efficiency:high_pressure_axe",
            shape=1,
            val=0.98,
        )
        self.add_input("electric_power", units="W", shape=n, val=np.nan)

        self.add_input(
            "data:propulsion:turboprop:design_point:turbine_entry_temperature",
            np.nan,
            units="K",
        )
        self.add_input(
            "data:propulsion:turboprop:design_point:power",
            np.nan,
            units="kW",
        )

        self.add_input(
            "settings:propulsion:turboprop:efficiency:gearbox",
            val=0.98,
        )
        self.add_input("settings:propulsion:turboprop:design_point:mach_exhaust", val=0.4)

        self.add_output(
            "fuel_mass_flow",
            units="kg/s",
            val=np.full(n, 0.06),
            shape=n,
        )
        self.add_output(
            "total_temperature_4",
            units="degK",
            val=np.full(n, 1350.0),
            shape=n,
        )
        self.add_output(
            "total_temperature_45",
            units="degK",
            val=np.full(n, 1000.0),
            shape=n,
        )
        self.add_output(
            "total_temperature_5",
            units="degK",
            val=np.full(n, 800.0),
            shape=n,
        )
        self.add_output(
            "total_pressure_45",
            units="Pa",
            val=np.full(n, 400000.0),
            shape=n,
        )
        self.add_output(
            "total_pressure_5",
            units="Pa",
            val=np.full(n, 400000.0),
            shape=n,
        )
        self.add_output(
            "air_mass_flow",
            units="kg/s",
            val=np.full(n, 2.5),
            shape=n,
        )

        self.declare_partials(
            of="fuel_mass_flow",
            wrt=[
                "combustion_energy",
                "cp_3",
                "cp_4",
                "total_temperature_3",
                "total_temperature_4",
                "fuel_air_ratio",
                "compressor_bleed_ratio",
                "cooling_bleed_ratio",
                "pressurization_bleed_ratio",
            ],
            method="exact",
        )
        self.declare_partials(
            of="total_temperature_4",
            wrt=[
                "data:propulsion:turboprop:design_point:turbine_entry_temperature",
                "total_temperature_4",
                "total_temperature_3",
                "fuel_air_ratio",
                "compressor_bleed_ratio",
                "cooling_bleed_ratio",
                "pressurization_bleed_ratio",
            ],
            method="exact",
        )
        self.declare_partials(
            of="total_temperature_45",
            wrt=[
                "data:propulsion:turboprop:design_point:turbine_entry_temperature",
                "total_temperature_2",
                "total_temperature_25",
                "total_temperature_3",
                "total_temperature_45",
                "cp_2",
                "cp_25",
                "cp_3",
                "cp_41",
                "cp_45",
                "fuel_air_ratio",
                "compressor_bleed_ratio",
                "pressurization_bleed_ratio",
                "air_mass_flow",
                "electric_power",
                "settings:propulsion:turboprop:efficiency:high_pressure_axe",
            ],
            method="exact",
        )
        self.declare_partials(
            of="total_pressure_45",
            wrt=[
                "total_pressure_4",
                "total_pressure_45",
                "total_temperature_45",
                "data:propulsion:turboprop:design_point:turbine_entry_temperature",
                "gamma_41",
                "eta_445",
            ],
            method="exact",
        )
        self.declare_partials(
            of="total_temperature_5",
            wrt=[
                "air_mass_flow",
                "total_temperature_45",
                "total_temperature_5",
                "cp_45",
                "cp_5",
                "fuel_air_ratio",
                "compressor_bleed_ratio",
                "pressurization_bleed_ratio",
                "data:propulsion:turboprop:design_point:power",
                "settings:propulsion:turboprop:efficiency:gearbox",
            ],
            method="exact",
        )
        self.declare_partials(
            of="total_pressure_5",
            wrt=[
                "total_temperature_5",
                "total_temperature_45",
                "total_pressure_5",
                "total_pressure_45",
                "gamma_45",
                "eta_455",
            ],
            method="exact",
        )
        self.declare_partials(
            of="air_mass_flow",
            wrt=[
                "total_pressure_5",
                "static_pressure_0",
                "gamma_5",
                "settings:propulsion:turboprop:design_point:mach_exhaust",
            ],
            method="exact",
        )

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        cp_2 = inputs["cp_2"]
        cp_25 = inputs["cp_25"]
        cp_3 = inputs["cp_3"]
        cp_4 = inputs["cp_4"]
        cp_41 = inputs["cp_41"]
        cp_45 = inputs["cp_45"]
        cp_5 = inputs["cp_5"]

        gamma_41 = inputs["gamma_41"]
        gamma_45 = inputs["gamma_45"]
        gamma_5 = inputs["gamma_5"]

        eta_445 = inputs["eta_445"]
        eta_455 = inputs["eta_455"]

        static_pressure = inputs["static_pressure_0"]

        total_temperature_2 = inputs["total_temperature_2"]
        total_temperature_25 = inputs["total_temperature_25"]
        total_temperature_3 = inputs["total_temperature_3"]
        total_temperature_41 = inputs[
            "data:propulsion:turboprop:design_point:turbine_entry_temperature"
        ]

        total_pressure_4 = inputs["total_pressure_4"]

        fuel_air_ratio = inputs["fuel_air_ratio"]
        compressor_bleed_ratio = inputs["compressor_bleed_ratio"]
        cooling_bleed_ratio = inputs["cooling_bleed_ratio"]
        pressurization_bleed_ratio = inputs["pressurization_bleed_ratio"]

        combustion_energy = inputs["combustion_energy"]

        mechanical_efficiency = inputs["settings:propulsion:turboprop:efficiency:high_pressure_axe"]
        electric_power = inputs["electric_power"]
        design_point_power = inputs["data:propulsion:turboprop:design_point:power"]
        eta_gearbox = inputs["settings:propulsion:turboprop:efficiency:gearbox"]
        exhaust_mach = inputs["settings:propulsion:turboprop:design_point:mach_exhaust"]

        total_temperature_4 = outputs["total_temperature_4"]
        total_temperature_45 = outputs["total_temperature_45"]
        total_pressure_45 = outputs["total_pressure_45"]
        total_temperature_5 = outputs["total_temperature_5"]
        air_mass_flow = outputs["air_mass_flow"]
        total_pressure_5 = outputs["total_pressure_5"]

        residuals["fuel_mass_flow"] = (cp_4 * total_temperature_4 - cp_3 * total_temperature_3) * (
            1
            + fuel_air_ratio
            - pressurization_bleed_ratio
            - cooling_bleed_ratio
            - compressor_bleed_ratio
        ) / 1000.0 - combustion_energy * fuel_air_ratio / 1000.0

        residuals["total_temperature_4"] = total_temperature_41 - (
            total_temperature_4
            * (
                1
                + fuel_air_ratio
                - pressurization_bleed_ratio
                - cooling_bleed_ratio
                - compressor_bleed_ratio
            )
            + total_temperature_3 * cooling_bleed_ratio
        ) / (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)

        residuals["total_temperature_45"] = (
            (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * (cp_41 * total_temperature_41 - cp_45 * total_temperature_45)
            * mechanical_efficiency
            - electric_power / air_mass_flow
            - (cp_3 * total_temperature_3 - cp_25 * total_temperature_25)
            * (1 - compressor_bleed_ratio)
            - (cp_25 * total_temperature_25 - cp_2 * total_temperature_2)
        )

        residuals["total_pressure_45"] = total_pressure_45 - total_pressure_4 * (
            total_temperature_45 / total_temperature_41
        ) ** (gamma_41 / (gamma_41 - 1.0) / eta_445)

        residuals["total_temperature_5"] = (
            air_mass_flow * 1000.0
            - (
                (
                    (design_point_power * 1000.0 / eta_gearbox)
                    / (cp_45 * total_temperature_45 - cp_5 * total_temperature_5)
                )
                / (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            )
            * 1000.0
        )

        residuals["total_pressure_5"] = total_temperature_5 - total_temperature_45 * (
            (total_pressure_5 / total_pressure_45) ** ((gamma_45 - 1.0) / gamma_45 * eta_455)
        )

        residuals["air_mass_flow"] = total_pressure_5 - (
            static_pressure
            * (1.0 + (gamma_5 - 1.0) / 2.0 * exhaust_mach ** 2) ** (gamma_5 / (gamma_5 - 1.0))
        )

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):

        n = self.options["number_of_points"]

        cp_2 = inputs["cp_2"]
        cp_25 = inputs["cp_25"]
        cp_3 = inputs["cp_3"]
        cp_4 = inputs["cp_4"]
        cp_41 = inputs["cp_41"]
        cp_45 = inputs["cp_45"]
        cp_5 = inputs["cp_5"]

        gamma_41 = inputs["gamma_41"]
        gamma_45 = inputs["gamma_45"]
        gamma_5 = inputs["gamma_5"]

        eta_445 = inputs["eta_445"]
        eta_455 = inputs["eta_455"]

        total_temperature_2 = inputs["total_temperature_2"]
        total_temperature_25 = inputs["total_temperature_25"]
        total_temperature_3 = inputs["total_temperature_3"]
        total_temperature_41 = inputs[
            "data:propulsion:turboprop:design_point:turbine_entry_temperature"
        ]

        static_pressure = inputs["static_pressure_0"]

        total_pressure_4 = inputs["total_pressure_4"]

        fuel_air_ratio = inputs["fuel_air_ratio"]
        compressor_bleed_ratio = inputs["compressor_bleed_ratio"]
        cooling_bleed_ratio = inputs["cooling_bleed_ratio"]
        pressurization_bleed_ratio = inputs["pressurization_bleed_ratio"]

        combustion_energy = inputs["combustion_energy"]

        design_point_power = inputs["data:propulsion:turboprop:design_point:power"]
        eta_gearbox = inputs["settings:propulsion:turboprop:efficiency:gearbox"]
        exhaust_mach = inputs["settings:propulsion:turboprop:design_point:mach_exhaust"]

        mechanical_efficiency = inputs["settings:propulsion:turboprop:efficiency:high_pressure_axe"]
        electric_power = inputs["electric_power"]

        total_temperature_4 = outputs["total_temperature_4"]
        total_temperature_45 = outputs["total_temperature_45"]
        total_pressure_45 = outputs["total_pressure_45"]
        air_mass_flow = outputs["air_mass_flow"]
        total_temperature_5 = outputs["total_temperature_5"]
        total_pressure_5 = outputs["total_pressure_5"]

        # -----------------------------------------------------------------------------------------#
        jacobian["fuel_mass_flow", "combustion_energy"] = -fuel_air_ratio / 1000.0
        jacobian["fuel_mass_flow", "cp_3"] = (
            -np.diag(
                total_temperature_3
                * (
                    1
                    + fuel_air_ratio
                    - pressurization_bleed_ratio
                    - cooling_bleed_ratio
                    - compressor_bleed_ratio
                )
            )
            / 1000.0
        )
        jacobian["fuel_mass_flow", "cp_4"] = (
            np.diag(
                total_temperature_4
                * (
                    1
                    + fuel_air_ratio
                    - pressurization_bleed_ratio
                    - cooling_bleed_ratio
                    - compressor_bleed_ratio
                )
            )
            / 1000.0
        )
        jacobian["fuel_mass_flow", "total_temperature_3"] = (
            -np.diag(
                cp_3
                * (
                    1
                    + fuel_air_ratio
                    - pressurization_bleed_ratio
                    - cooling_bleed_ratio
                    - compressor_bleed_ratio
                )
            )
            / 1000.0
        )
        jacobian["fuel_mass_flow", "total_temperature_4"] = (
            np.diag(
                cp_4
                * (
                    1
                    + fuel_air_ratio
                    - pressurization_bleed_ratio
                    - cooling_bleed_ratio
                    - compressor_bleed_ratio
                )
            )
            / 1000.0
        )
        jacobian["fuel_mass_flow", "fuel_air_ratio"] = (
            np.diag((cp_4 * total_temperature_4 - cp_3 * total_temperature_3) - combustion_energy)
            / 1000.0
        )
        jacobian["fuel_mass_flow", "compressor_bleed_ratio"] = (
            -np.diag((cp_4 * total_temperature_4 - cp_3 * total_temperature_3)) / 1000.0
        )
        jacobian["fuel_mass_flow", "cooling_bleed_ratio"] = (
            -np.diag((cp_4 * total_temperature_4 - cp_3 * total_temperature_3)) / 1000.0
        )
        jacobian["fuel_mass_flow", "pressurization_bleed_ratio"] = (
            -np.diag((cp_4 * total_temperature_4 - cp_3 * total_temperature_3)) / 1000.0
        )

        # -----------------------------------------------------------------------------------------#
        jacobian[
            "total_temperature_4",
            "data:propulsion:turboprop:design_point:turbine_entry_temperature",
        ] = np.eye(n)
        jacobian["total_temperature_4", "total_temperature_4"] = -np.diag(
            (
                1
                + fuel_air_ratio
                - pressurization_bleed_ratio
                - cooling_bleed_ratio
                - compressor_bleed_ratio
            )
            / (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
        )
        jacobian["total_temperature_4", "total_temperature_3"] = -np.diag(
            cooling_bleed_ratio
            / (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
        )
        jacobian["total_temperature_4", "fuel_air_ratio"] = np.diag(
            (total_temperature_3 - total_temperature_4)
            * cooling_bleed_ratio
            / (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio) ** 2.0
        )
        jacobian["total_temperature_4", "pressurization_bleed_ratio"] = -np.diag(
            (total_temperature_3 - total_temperature_4)
            * cooling_bleed_ratio
            / (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio) ** 2.0
        )
        jacobian["total_temperature_4", "cooling_bleed_ratio"] = np.diag(
            (total_temperature_4 - total_temperature_3)
            / (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
        )
        jacobian["total_temperature_4", "compressor_bleed_ratio"] = -np.diag(
            (total_temperature_3 - total_temperature_4)
            * cooling_bleed_ratio
            / (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio) ** 2.0
        )

        # -----------------------------------------------------------------------------------------#

        jacobian[
            "total_temperature_45",
            "data:propulsion:turboprop:design_point:turbine_entry_temperature",
        ] = np.diag(
            (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * cp_41
            * mechanical_efficiency
        )
        jacobian["total_temperature_45", "total_temperature_2"] = np.diag(cp_2)
        jacobian["total_temperature_45", "total_temperature_25"] = -np.diag(
            cp_25 * compressor_bleed_ratio
        )
        jacobian["total_temperature_45", "total_temperature_3"] = -np.diag(
            cp_3 * (1.0 - compressor_bleed_ratio)
        )
        jacobian["total_temperature_45", "total_temperature_45"] = -np.diag(
            (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * cp_45
            * mechanical_efficiency
        )
        jacobian["total_temperature_45", "cp_2"] = np.diag(total_temperature_2)
        jacobian["total_temperature_45", "cp_25"] = -np.diag(
            total_temperature_25 * compressor_bleed_ratio
        )
        jacobian["total_temperature_45", "cp_3"] = -np.diag(
            total_temperature_3 * (1.0 - compressor_bleed_ratio)
        )
        jacobian["total_temperature_45", "cp_41"] = np.diag(
            (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * total_temperature_41
            * mechanical_efficiency
        )
        jacobian["total_temperature_45", "cp_45"] = -np.diag(
            (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * total_temperature_45
            * mechanical_efficiency
        )
        jacobian["total_temperature_45", "fuel_air_ratio"] = np.diag(
            (cp_41 * total_temperature_41 - cp_45 * total_temperature_45) * mechanical_efficiency
        )
        jacobian["total_temperature_45", "compressor_bleed_ratio"] = np.diag(
            -(cp_41 * total_temperature_41 - cp_45 * total_temperature_45) * mechanical_efficiency
            + (cp_3 * total_temperature_3 - cp_25 * total_temperature_25)
        )
        jacobian["total_temperature_45", "pressurization_bleed_ratio"] = -np.diag(
            (cp_41 * total_temperature_41 - cp_45 * total_temperature_45) * mechanical_efficiency
        )
        jacobian["total_temperature_45", "air_mass_flow"] = electric_power / air_mass_flow ** 2.0
        jacobian["total_temperature_45", "electric_power"] = -1.0 / air_mass_flow
        jacobian[
            "total_temperature_45",
            "settings:propulsion:turboprop:efficiency:high_pressure_axe",
        ] = np.diag(
            (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * (cp_41 * total_temperature_41 - cp_45 * total_temperature_45)
        )

        # -----------------------------------------------------------------------------------------#
        jacobian["total_pressure_45", "total_pressure_45"] = np.eye(n)
        jacobian["total_pressure_45", "total_pressure_4"] = -np.diag(
            (total_temperature_45 / total_temperature_41) ** (gamma_41 / (gamma_41 - 1.0) / eta_445)
        )
        jacobian["total_pressure_45", "total_temperature_45"] = -np.diag(
            total_pressure_4
            * (gamma_41 / (gamma_41 - 1.0) / eta_445)
            * (total_temperature_45 / total_temperature_41)
            ** (gamma_41 / (gamma_41 - 1.0) / eta_445 - 1.0)
            / total_temperature_41
        )
        jacobian[
            "total_pressure_45",
            "data:propulsion:turboprop:design_point:turbine_entry_temperature",
        ] = np.diag(
            total_pressure_4
            * (gamma_41 / (gamma_41 - 1.0) / eta_445)
            * (total_temperature_45 / total_temperature_41)
            ** (gamma_41 / (gamma_41 - 1.0) / eta_445 - 1.0)
            * total_temperature_45
            / total_temperature_41 ** 2.0
        )
        jacobian["total_pressure_45", "gamma_41"] = np.diag(
            total_pressure_4
            * np.log(total_temperature_45 / total_temperature_41)
            * (total_temperature_45 / total_temperature_41) ** (gamma_41 / (gamma_41 - 1) / eta_445)
            / (gamma_41 - 1) ** 2.0
            / eta_445
        )
        jacobian["total_pressure_45", "eta_445"] = (
            total_pressure_4
            * np.log(total_temperature_45 / total_temperature_41)
            * (total_temperature_45 / total_temperature_41) ** (gamma_41 / (gamma_41 - 1) / eta_445)
            * gamma_41
            / (gamma_41 - 1)
            / eta_445 ** 2.0
        )

        # -----------------------------------------------------------------------------------------#

        jacobian["total_temperature_5", "air_mass_flow"] = np.eye(n) * 1000.0
        jacobian["total_temperature_5", "total_temperature_45"] = 1000.0 * np.diag(
            (design_point_power * 1000.0 / eta_gearbox)
            / (cp_45 * total_temperature_45 - cp_5 * total_temperature_5) ** 2.0
            / (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * cp_45
        )
        jacobian["total_temperature_5", "total_temperature_5"] = -1000.0 * np.diag(
            (design_point_power * 1000.0 / eta_gearbox)
            / (cp_45 * total_temperature_45 - cp_5 * total_temperature_5) ** 2.0
            / (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * cp_5
        )
        jacobian["total_temperature_5", "cp_45"] = 1000.0 * np.diag(
            (design_point_power * 1000.0 / eta_gearbox)
            / (cp_45 * total_temperature_45 - cp_5 * total_temperature_5) ** 2.0
            / (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * total_temperature_45
        )
        jacobian["total_temperature_5", "cp_5"] = -1000.0 * np.diag(
            (design_point_power * 1000.0 / eta_gearbox)
            / (cp_45 * total_temperature_45 - cp_5 * total_temperature_5) ** 2.0
            / (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * total_temperature_5
        )
        jacobian["total_temperature_5", "fuel_air_ratio"] = 1000.0 * np.diag(
            (design_point_power * 1000.0 / eta_gearbox)
            / (cp_45 * total_temperature_45 - cp_5 * total_temperature_5)
            / (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio) ** 2.0
        )
        jacobian["total_temperature_5", "compressor_bleed_ratio"] = -1000.0 * np.diag(
            (design_point_power * 1000.0 / eta_gearbox)
            / (cp_45 * total_temperature_45 - cp_5 * total_temperature_5)
            / (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio) ** 2.0
        )
        jacobian["total_temperature_5", "pressurization_bleed_ratio"] = -1000.0 * np.diag(
            (design_point_power * 1000.0 / eta_gearbox)
            / (cp_45 * total_temperature_45 - cp_5 * total_temperature_5)
            / (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio) ** 2.0
        )
        jacobian["total_temperature_5", "data:propulsion:turboprop:design_point:power"] = (
            -(
                (
                    (1000.0 / eta_gearbox)
                    / (cp_45 * total_temperature_45 - cp_5 * total_temperature_5)
                )
                / (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            )
            * 1000.0
        )
        jacobian["total_temperature_5", "settings:propulsion:turboprop:efficiency:gearbox"] = (
            (
                (design_point_power * 1000.0 / eta_gearbox ** 2.0)
                / (cp_45 * total_temperature_45 - cp_5 * total_temperature_5)
            )
            / (1 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
        ) * 1000.0

        # ---------------------------------------------------------------------------------------- #

        jacobian["total_pressure_5", "total_temperature_5"] = np.eye(n)
        jacobian["total_pressure_5", "total_temperature_45"] = -np.diag(
            (total_pressure_5 / total_pressure_45) ** ((gamma_45 - 1.0) / gamma_45 * eta_455)
        )
        jacobian["total_pressure_5", "total_pressure_5"] = -np.diag(
            total_temperature_45
            * ((gamma_45 - 1.0) / gamma_45 * eta_455)
            * (total_pressure_5 / total_pressure_45)
            ** ((gamma_45 - 1.0) / gamma_45 * eta_455 - 1.0)
            / total_pressure_45
        )
        jacobian["total_pressure_5", "total_pressure_45"] = np.diag(
            total_temperature_45
            * ((gamma_45 - 1.0) / gamma_45 * eta_455)
            * (total_pressure_5 / total_pressure_45)
            ** ((gamma_45 - 1.0) / gamma_45 * eta_455 - 1.0)
            * total_pressure_5
            / total_pressure_45 ** 2.0
        )
        jacobian["total_pressure_5", "gamma_45"] = np.diag(
            -total_temperature_45
            * ((total_pressure_5 / total_pressure_45) ** ((gamma_45 - 1.0) / gamma_45 * eta_455))
            * np.log(total_pressure_5 / total_pressure_45)
            * eta_455
            / gamma_45 ** 2.0
        )
        jacobian["total_pressure_5", "eta_455"] = np.diag(
            -total_temperature_45
            * ((total_pressure_5 / total_pressure_45) ** ((gamma_45 - 1.0) / gamma_45 * eta_455))
            * np.log(total_pressure_5 / total_pressure_45)
            * (gamma_45 - 1.0)
            / gamma_45
        )

        # ---------------------------------------------------------------------------------------- #

        jacobian["air_mass_flow", "total_pressure_5"] = np.eye(n)
        jacobian["air_mass_flow", "static_pressure_0"] = -np.diag(
            (1 + (gamma_5 - 1) / 2 * exhaust_mach ** 2) ** (gamma_5 / (gamma_5 - 1.0))
        )
        jacobian["air_mass_flow", "gamma_5"] = -np.diag(
            (
                static_pressure
                * (1.0 + 1.0 / 2.0 * exhaust_mach ** 2 * (gamma_5 - 1.0))
                ** (gamma_5 / (gamma_5 - 1.0))
                * (
                    exhaust_mach ** 2 * (gamma_5 - 1.0) * gamma_5
                    + (-2.0 - exhaust_mach ** 2.0 * (gamma_5 - 1.0))
                    * np.log(1.0 + 1.0 / 2.0 * exhaust_mach ** 2.0 * (gamma_5 - 1))
                )
            )
            / ((2 + exhaust_mach ** 2.0 * (gamma_5 - 1.0)) * (gamma_5 - 1) ** 2.0)
        )
        jacobian["air_mass_flow", "settings:propulsion:turboprop:design_point:mach_exhaust"] = (
            -static_pressure
            * gamma_5
            * exhaust_mach
            * (1.0 + (gamma_5 - 1.0) / 2.0 * exhaust_mach ** 2) ** (1.0 / (gamma_5 - 1.0))
        )

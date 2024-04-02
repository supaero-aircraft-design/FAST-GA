import numpy as np
import openmdao.api as om


class ThermodynamicEquilibrium(om.ImplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("combustion_energy", shape=1, val=0.95 * 43.260e6, units="J/kg")

        self.add_input("fuel_mass_flow", units="kg/s", val=np.nan, shape=n)
        self.add_input("electric_power", units="W", shape=n, val=np.nan)

        self.add_input("fuel_air_ratio", shape=n, val=np.nan)
        self.add_input("compressor_bleed_ratio", shape=n, val=np.nan)
        self.add_input("cooling_bleed_ratio", shape=n, val=np.nan)
        self.add_input("pressurization_bleed_ratio", shape=n, val=np.nan)

        self.add_input("total_temperature_2", units="K", shape=n, val=np.nan)
        self.add_input("total_temperature_3", units="K", shape=n, val=np.nan)
        self.add_input("total_temperature_4", units="K", shape=n, val=np.nan)
        self.add_input("total_pressure_25", units="Pa", shape=n, val=np.nan)
        self.add_input("total_pressure_3", units="Pa", shape=n, val=np.nan)
        self.add_input("total_pressure_41", units="Pa", shape=n, val=np.nan)
        self.add_input("cp_2", shape=n, val=np.nan)
        self.add_input("cp_25", shape=n, val=np.nan)
        self.add_input("cp_3", shape=n, val=np.nan)
        self.add_input("cp_4", shape=n, val=np.nan)
        self.add_input("cp_41", shape=n, val=np.nan)
        self.add_input("cp_45", shape=n, val=np.nan)
        self.add_input("gamma_41", shape=n, val=np.nan)
        self.add_input("gamma_25", shape=n, val=np.nan)

        self.add_input("opr_1", shape=n, val=np.nan)
        self.add_input("opr_2", shape=n, val=np.nan)

        self.add_input("eta_253", shape=1, val=1.0)
        self.add_input(
            "settings:propulsion:turboprop:efficiency:high_pressure_axe",
            shape=1,
            val=0.98,
        )

        self.add_input(
            "data:propulsion:turboprop:section:41",
            units="m**2",
            shape=1,
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:turboprop:design_point:alpha",
            shape=1,
            val=0.8,
        )

        self.add_output(
            "air_mass_flow",
            units="kg/s",
            val=np.full(n, 2.5),
            shape=n,
        )
        self.add_output(
            "total_temperature_41",
            units="K",
            val=np.full(n, 1160.0),
            shape=n,
        )
        self.add_output(
            "total_temperature_25",
            units="K",
            val=np.full(n, 380.0),
            shape=n,
        )

        self.declare_partials(
            of="air_mass_flow",
            wrt=[
                "combustion_energy",
                "fuel_mass_flow",
                "fuel_air_ratio",
                "compressor_bleed_ratio",
                "cooling_bleed_ratio",
                "pressurization_bleed_ratio",
                "total_temperature_3",
                "total_temperature_4",
                "cp_3",
                "cp_4",
                "air_mass_flow",
            ],
            method="exact",
        )
        self.declare_partials(
            of="total_temperature_41",
            wrt=[
                "data:propulsion:turboprop:section:41",
                "air_mass_flow",
                "fuel_air_ratio",
                "compressor_bleed_ratio",
                "pressurization_bleed_ratio",
                "total_temperature_41",
                "total_pressure_41",
            ],
            method="exact",
        )
        self.declare_partials(
            of="total_temperature_41",
            wrt="gamma_41",
            method="fd",
            step=1e-4,
        )
        self.declare_partials(
            of="total_temperature_25",
            wrt=[
                "air_mass_flow",
                "electric_power",
                "total_temperature_2",
                "total_temperature_25",
                "total_temperature_3",
                "total_temperature_41",
                "fuel_air_ratio",
                "compressor_bleed_ratio",
                "pressurization_bleed_ratio",
                "settings:propulsion:turboprop:efficiency:high_pressure_axe",
                "cp_2",
                "cp_25",
                "cp_3",
                "cp_41",
                "cp_45",
                "data:propulsion:turboprop:design_point:alpha",
            ],
            method="exact",
        )

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        r_g = 287.0  # Perfect gas constant

        combustion_energy = inputs["combustion_energy"]

        fuel_mass_flow = inputs["fuel_mass_flow"]
        electric_power = inputs["electric_power"]

        fuel_air_ratio = inputs["fuel_air_ratio"]
        compressor_bleed_ratio = inputs["compressor_bleed_ratio"]
        cooling_bleed_ratio = inputs["cooling_bleed_ratio"]
        pressurization_bleed_ratio = inputs["pressurization_bleed_ratio"]

        total_temperature_2 = inputs["total_temperature_2"]
        total_temperature_3 = inputs["total_temperature_3"]
        total_temperature_4 = inputs["total_temperature_4"]
        total_pressure_41 = inputs["total_pressure_41"]
        cp_2 = inputs["cp_2"]
        cp_25 = inputs["cp_25"]
        cp_3 = inputs["cp_3"]
        cp_4 = inputs["cp_4"]
        cp_41 = inputs["cp_41"]
        cp_45 = inputs["cp_45"]
        gamma_41 = inputs["gamma_41"]

        a_41 = inputs["data:propulsion:turboprop:section:41"]
        alpha = inputs["data:propulsion:turboprop:design_point:alpha"]

        mechanical_efficiency = inputs["settings:propulsion:turboprop:efficiency:high_pressure_axe"]

        air_mass_flow = outputs["air_mass_flow"]
        total_temperature_41 = outputs["total_temperature_41"]
        total_temperature_25 = outputs["total_temperature_25"]

        f_gamma_41 = np.sqrt(gamma_41) * (2.0 / (gamma_41 + 1.0)) ** (
            (gamma_41 + 1) / 2.0 / (gamma_41 - 1.0)
        )

        residuals["air_mass_flow"] = 1.0 - air_mass_flow * (
            1.0
            + fuel_air_ratio
            - pressurization_bleed_ratio
            - cooling_bleed_ratio
            - compressor_bleed_ratio
        ) * (cp_4 * total_temperature_4 - cp_3 * total_temperature_3) / (
            fuel_mass_flow * combustion_energy
        )

        residuals["total_temperature_41"] = (
            1.0
            - air_mass_flow
            / a_41
            * (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * np.sqrt(total_temperature_41 * r_g)
            / total_pressure_41
            / f_gamma_41
        )

        residuals["total_temperature_25"] = (
            1.0
            - total_temperature_2
            / total_temperature_3
            * cp_2
            / cp_3
            / (1.0 - compressor_bleed_ratio)
            - mechanical_efficiency
            * (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            / (1.0 - compressor_bleed_ratio)
            * (cp_41 - cp_45 * alpha)
            / cp_3
            * total_temperature_41
            / total_temperature_3
            + electric_power
            / cp_3
            / total_temperature_3
            / air_mass_flow
            / (1.0 - compressor_bleed_ratio)
            + total_temperature_25
            / total_temperature_3
            * cp_25
            / cp_3
            * compressor_bleed_ratio
            / (1.0 - compressor_bleed_ratio)
        )

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):

        r_g = 287.0  # Perfect gas constant

        combustion_energy = inputs["combustion_energy"]

        fuel_mass_flow = inputs["fuel_mass_flow"]
        electric_power = inputs["electric_power"]

        fuel_air_ratio = inputs["fuel_air_ratio"]
        compressor_bleed_ratio = inputs["compressor_bleed_ratio"]
        cooling_bleed_ratio = inputs["cooling_bleed_ratio"]
        pressurization_bleed_ratio = inputs["pressurization_bleed_ratio"]

        total_temperature_2 = inputs["total_temperature_2"]
        total_temperature_3 = inputs["total_temperature_3"]
        total_temperature_4 = inputs["total_temperature_4"]
        total_pressure_41 = inputs["total_pressure_41"]
        cp_2 = inputs["cp_2"]
        cp_25 = inputs["cp_25"]
        cp_3 = inputs["cp_3"]
        cp_4 = inputs["cp_4"]
        cp_41 = inputs["cp_41"]
        cp_45 = inputs["cp_45"]
        gamma_41 = inputs["gamma_41"]

        alpha = inputs["data:propulsion:turboprop:design_point:alpha"]

        mechanical_efficiency = inputs["settings:propulsion:turboprop:efficiency:high_pressure_axe"]

        a_41 = inputs["data:propulsion:turboprop:section:41"]

        air_mass_flow = outputs["air_mass_flow"]
        total_temperature_41 = outputs["total_temperature_41"]
        total_temperature_25 = outputs["total_temperature_25"]

        f_gamma_41 = np.sqrt(gamma_41) * (2.0 / (gamma_41 + 1.0)) ** (
            (gamma_41 + 1) / 2.0 / (gamma_41 - 1.0)
        )

        # ------------------ Derivatives wrt air mass flow residuals ------------------ #

        jacobian["air_mass_flow", "combustion_energy"] = (
            air_mass_flow
            * (
                1.0
                + fuel_air_ratio
                - pressurization_bleed_ratio
                - cooling_bleed_ratio
                - compressor_bleed_ratio
            )
            * (cp_4 * total_temperature_4 - cp_3 * total_temperature_3)
            / (fuel_mass_flow * combustion_energy ** 2.0)
        )
        jacobian["air_mass_flow", "fuel_mass_flow"] = np.diag(
            air_mass_flow
            * (
                1.0
                + fuel_air_ratio
                - pressurization_bleed_ratio
                - cooling_bleed_ratio
                - compressor_bleed_ratio
            )
            * (cp_4 * total_temperature_4 - cp_3 * total_temperature_3)
            / (fuel_mass_flow ** 2.0 * combustion_energy)
        )

        jacobian["air_mass_flow", "fuel_air_ratio"] = -np.diag(
            air_mass_flow
            * (cp_4 * total_temperature_4 - cp_3 * total_temperature_3)
            / (fuel_mass_flow * combustion_energy)
        )
        jacobian["air_mass_flow", "compressor_bleed_ratio"] = np.diag(
            air_mass_flow
            * (cp_4 * total_temperature_4 - cp_3 * total_temperature_3)
            / (fuel_mass_flow * combustion_energy)
        )
        jacobian["air_mass_flow", "cooling_bleed_ratio"] = np.diag(
            air_mass_flow
            * (cp_4 * total_temperature_4 - cp_3 * total_temperature_3)
            / (fuel_mass_flow * combustion_energy)
        )
        jacobian["air_mass_flow", "pressurization_bleed_ratio"] = np.diag(
            air_mass_flow
            * (cp_4 * total_temperature_4 - cp_3 * total_temperature_3)
            / (fuel_mass_flow * combustion_energy)
        )

        jacobian["air_mass_flow", "total_temperature_3"] = np.diag(
            air_mass_flow
            * (
                1.0
                + fuel_air_ratio
                - pressurization_bleed_ratio
                - cooling_bleed_ratio
                - compressor_bleed_ratio
            )
            * cp_3
            / (fuel_mass_flow * combustion_energy)
        )
        jacobian["air_mass_flow", "cp_3"] = np.diag(
            air_mass_flow
            * (
                1.0
                + fuel_air_ratio
                - pressurization_bleed_ratio
                - cooling_bleed_ratio
                - compressor_bleed_ratio
            )
            * total_temperature_3
            / (fuel_mass_flow * combustion_energy)
        )

        jacobian["air_mass_flow", "total_temperature_4"] = -np.diag(
            air_mass_flow
            * (
                1.0
                + fuel_air_ratio
                - pressurization_bleed_ratio
                - cooling_bleed_ratio
                - compressor_bleed_ratio
            )
            * cp_4
            / (fuel_mass_flow * combustion_energy)
        )
        jacobian["air_mass_flow", "cp_4"] = -np.diag(
            air_mass_flow
            * (
                1.0
                + fuel_air_ratio
                - pressurization_bleed_ratio
                - cooling_bleed_ratio
                - compressor_bleed_ratio
            )
            * total_temperature_4
            / (fuel_mass_flow * combustion_energy)
        )
        jacobian["air_mass_flow", "air_mass_flow"] = -np.diag(
            (
                1.0
                + fuel_air_ratio
                - pressurization_bleed_ratio
                - cooling_bleed_ratio
                - compressor_bleed_ratio
            )
            * (cp_4 * total_temperature_4 - cp_3 * total_temperature_3)
            / (fuel_mass_flow * combustion_energy)
        )

        # ------------------ Derivatives wrt total temperature 41 residuals ------------------ #

        jacobian["total_temperature_41", "data:propulsion:turboprop:section:41"] = (
            air_mass_flow
            / a_41 ** 2.0
            * (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * np.sqrt(total_temperature_41 * r_g)
            / total_pressure_41
            / f_gamma_41
        )
        jacobian["total_temperature_41", "air_mass_flow"] = np.diag(
            -(1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * np.sqrt(total_temperature_41 * r_g)
            / total_pressure_41
            / f_gamma_41
            / a_41
        )
        jacobian["total_temperature_41", "fuel_air_ratio"] = -np.diag(
            air_mass_flow
            * np.sqrt(total_temperature_41 * r_g)
            / total_pressure_41
            / f_gamma_41
            / a_41
        )
        jacobian["total_temperature_41", "pressurization_bleed_ratio"] = np.diag(
            air_mass_flow
            * np.sqrt(total_temperature_41 * r_g)
            / total_pressure_41
            / f_gamma_41
            / a_41
        )
        jacobian["total_temperature_41", "compressor_bleed_ratio"] = np.diag(
            air_mass_flow
            * np.sqrt(total_temperature_41 * r_g)
            / total_pressure_41
            / f_gamma_41
            / a_41
        )
        jacobian["total_temperature_41", "total_temperature_41"] = -np.diag(
            air_mass_flow
            * (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * np.sqrt(r_g / total_temperature_41)
            / total_pressure_41
            / f_gamma_41
            / 2.0
            / a_41
        )
        jacobian["total_temperature_41", "total_pressure_41"] = np.diag(
            air_mass_flow
            * (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * np.sqrt(total_temperature_41 * r_g)
            / total_pressure_41 ** 2.0
            / f_gamma_41
            / a_41
        )

        # ------------------ Derivatives wrt total temperature 25 residuals ------------------ #

        jacobian["total_temperature_25", "air_mass_flow"] = -np.diag(
            electric_power
            / (total_temperature_3 * (1.0 - compressor_bleed_ratio) * cp_3 * air_mass_flow ** 2.0)
        )
        jacobian["total_temperature_25", "electric_power"] = np.diag(
            1.0 / (total_temperature_3 * (1.0 - compressor_bleed_ratio) * cp_3 * air_mass_flow)
        )
        jacobian["total_temperature_25", "total_temperature_2"] = -np.diag(
            cp_2
            * air_mass_flow
            / (total_temperature_3 * (1.0 - compressor_bleed_ratio) * cp_3 * air_mass_flow)
        )
        jacobian["total_temperature_25", "total_temperature_25"] = np.diag(
            cp_25
            * compressor_bleed_ratio
            * air_mass_flow
            / (total_temperature_3 * (1.0 - compressor_bleed_ratio) * cp_3 * air_mass_flow)
        )
        jacobian["total_temperature_25", "total_temperature_3"] = -np.diag(
            (
                -total_temperature_2 * cp_2 * air_mass_flow
                - mechanical_efficiency
                * (1.0 + fuel_air_ratio - compressor_bleed_ratio - pressurization_bleed_ratio)
                * (cp_41 - cp_45 * alpha)
                * total_temperature_41
                * air_mass_flow
                + electric_power
                + total_temperature_25 * cp_25 * compressor_bleed_ratio * air_mass_flow
            )
            / (total_temperature_3 ** 2.0 * (1.0 - compressor_bleed_ratio) * cp_3 * air_mass_flow)
        )
        jacobian["total_temperature_25", "total_temperature_41"] = -np.diag(
            (1.0 + fuel_air_ratio - compressor_bleed_ratio - pressurization_bleed_ratio)
            * (cp_41 - cp_45 * alpha)
            * mechanical_efficiency
            * air_mass_flow
            / (total_temperature_3 * (1.0 - compressor_bleed_ratio) * cp_3 * air_mass_flow)
        )
        jacobian["total_temperature_25", "fuel_air_ratio"] = -np.diag(
            mechanical_efficiency
            * (cp_41 - cp_45 * alpha)
            * total_temperature_41
            * air_mass_flow
            / (total_temperature_3 * (1.0 - compressor_bleed_ratio) * cp_3 * air_mass_flow)
        )
        jacobian["total_temperature_25", "compressor_bleed_ratio"] = np.diag(
            -total_temperature_2
            / total_temperature_3
            * cp_2
            / cp_3
            / (1.0 - compressor_bleed_ratio) ** 2.0
            - mechanical_efficiency
            * total_temperature_41
            / total_temperature_3
            * (cp_41 - cp_45 * alpha)
            / cp_3
            * (fuel_air_ratio - pressurization_bleed_ratio)
            / (1.0 - compressor_bleed_ratio) ** 2.0
            + electric_power
            / total_temperature_3
            / cp_3
            / air_mass_flow
            / (1.0 - compressor_bleed_ratio) ** 2.0
            + total_temperature_25
            / total_temperature_3
            * cp_25
            / cp_3
            / (1.0 - compressor_bleed_ratio) ** 2.0
        )
        jacobian["total_temperature_25", "pressurization_bleed_ratio"] = np.diag(
            mechanical_efficiency
            * (cp_41 - cp_45 * alpha)
            * total_temperature_41
            * air_mass_flow
            / (total_temperature_3 * (1.0 - compressor_bleed_ratio) * cp_3 * air_mass_flow)
        )
        jacobian[
            "total_temperature_25",
            "settings:propulsion:turboprop:efficiency:high_pressure_axe",
        ] = -(
            (1.0 + fuel_air_ratio - compressor_bleed_ratio - pressurization_bleed_ratio)
            * (cp_41 - cp_45 * alpha)
            * total_temperature_41
            * air_mass_flow
            / (total_temperature_3 * (1.0 - compressor_bleed_ratio) * cp_3 * air_mass_flow)
        )
        jacobian["total_temperature_25", "cp_2"] = -np.diag(
            total_temperature_2
            * air_mass_flow
            / (total_temperature_3 * (1.0 - compressor_bleed_ratio) * cp_3 * air_mass_flow)
        )
        jacobian["total_temperature_25", "cp_25"] = np.diag(
            total_temperature_25
            * compressor_bleed_ratio
            * air_mass_flow
            / (total_temperature_3 * (1.0 - compressor_bleed_ratio) * cp_3 * air_mass_flow)
        )
        jacobian["total_temperature_25", "cp_3"] = -np.diag(
            (
                -total_temperature_2 * cp_2 * air_mass_flow
                - mechanical_efficiency
                * (1.0 + fuel_air_ratio - compressor_bleed_ratio - pressurization_bleed_ratio)
                * (cp_41 - cp_45 * alpha)
                * total_temperature_41
                * air_mass_flow
                + electric_power
                + total_temperature_25 * cp_25 * compressor_bleed_ratio * air_mass_flow
            )
            / (total_temperature_3 * (1.0 - compressor_bleed_ratio) * cp_3 ** 2.0 * air_mass_flow)
        )
        jacobian["total_temperature_25", "cp_41"] = -mechanical_efficiency * np.diag(
            (1.0 + fuel_air_ratio - compressor_bleed_ratio - pressurization_bleed_ratio)
            * total_temperature_41
            * air_mass_flow
            / (total_temperature_3 * (1.0 - compressor_bleed_ratio) * cp_3 * air_mass_flow)
        )
        jacobian["total_temperature_25", "cp_45"] = mechanical_efficiency * np.diag(
            (1.0 + fuel_air_ratio - compressor_bleed_ratio - pressurization_bleed_ratio)
            * total_temperature_41
            * air_mass_flow
            * alpha
            / (total_temperature_3 * (1.0 - compressor_bleed_ratio) * cp_3 * air_mass_flow)
        )
        jacobian["total_temperature_25", "data:propulsion:turboprop:design_point:alpha"] = (
            mechanical_efficiency
            * (1.0 + fuel_air_ratio - compressor_bleed_ratio - pressurization_bleed_ratio)
            * cp_45
            * total_temperature_41
            * air_mass_flow
        ) / (total_temperature_3 * (1.0 - compressor_bleed_ratio) * cp_3 * air_mass_flow)

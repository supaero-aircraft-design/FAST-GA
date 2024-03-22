import numpy as np
import openmdao.api as om


class Station8Mach(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("gamma_5", shape=n, val=np.nan)
        self.add_input(
            "data:propulsion:turboprop:section:45",
            units="m**2",
            shape=1,
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:turboprop:section:8",
            units="m**2",
            shape=1,
            val=np.nan,
        )
        self.add_input("total_pressure_45", units="Pa", shape=n, val=np.nan)
        self.add_input("static_pressure_0", units="Pa", shape=n, val=np.nan)

        self.add_output("mach_8", shape=n, val=0.5)

        self.declare_partials(
            of="mach_8",
            wrt=[
                "data:propulsion:turboprop:section:45",
                "data:propulsion:turboprop:section:8",
                "total_pressure_45",
                "static_pressure_0",
            ],
            method="exact",
        )
        self.declare_partials(
            of="mach_8",
            wrt="gamma_5",
            method="fd",
            step=1e-4,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        static_pressure_0 = inputs["static_pressure_0"]
        total_pressure_45 = inputs["total_pressure_45"]
        a_45 = inputs["data:propulsion:turboprop:section:45"]
        a_8 = inputs["data:propulsion:turboprop:section:8"]
        gamma_5 = inputs["gamma_5"]

        f_gamma_5 = (2.0 / (gamma_5 + 1.0)) ** ((gamma_5 + 1) / 2.0 / (gamma_5 - 1.0))

        outputs["mach_8"] = (
            f_gamma_5
            * a_45
            / a_8
            * (total_pressure_45 / static_pressure_0) ** ((gamma_5 + 1) / 2 / gamma_5)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        static_pressure_0 = inputs["static_pressure_0"]
        total_pressure_45 = inputs["total_pressure_45"]
        a_45 = inputs["data:propulsion:turboprop:section:45"]
        a_8 = inputs["data:propulsion:turboprop:section:8"]
        gamma_5 = inputs["gamma_5"]

        f_gamma_5 = (2.0 / (gamma_5 + 1.0)) ** ((gamma_5 + 1) / 2.0 / (gamma_5 - 1.0))

        partials["mach_8", "static_pressure_0"] = -np.diag(
            f_gamma_5
            * a_45
            / a_8
            * ((gamma_5 + 1) / 2 / gamma_5)
            * (total_pressure_45 / static_pressure_0) ** ((gamma_5 + 1) / 2 / gamma_5 - 1.0)
            * total_pressure_45
            / static_pressure_0 ** 2.0
        )
        partials["mach_8", "total_pressure_45"] = np.diag(
            f_gamma_5
            * a_45
            / a_8
            * ((gamma_5 + 1) / 2 / gamma_5)
            * (total_pressure_45 / static_pressure_0) ** ((gamma_5 + 1) / 2 / gamma_5 - 1.0)
            / static_pressure_0
        )
        partials["mach_8", "data:propulsion:turboprop:section:45"] = (
            f_gamma_5
            / a_8
            * (total_pressure_45 / static_pressure_0) ** ((gamma_5 + 1) / 2 / gamma_5)
        )
        partials["mach_8", "data:propulsion:turboprop:section:8"] = -(
            f_gamma_5
            * a_45
            / a_8 ** 2.0
            * (total_pressure_45 / static_pressure_0) ** ((gamma_5 + 1) / 2 / gamma_5)
        )


class Station8Temperature(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("gamma_5", shape=n, val=np.nan)
        self.add_input("mach_8", shape=n, val=np.nan)
        self.add_input("total_temperature_5", units="K", shape=n, val=np.nan)

        self.add_output("static_temperature_8", units="K", shape=n, val=3e2)

        self.declare_partials(of="static_temperature_8", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        gamma_5 = inputs["gamma_5"]
        mach_8 = inputs["mach_8"]
        total_temperature_5 = inputs["total_temperature_5"]

        static_temperature_8 = total_temperature_5 / (1.0 + (gamma_5 - 1.0) / 2 * mach_8 ** 2.0)

        outputs["static_temperature_8"] = static_temperature_8

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        gamma_5 = inputs["gamma_5"]
        mach_8 = inputs["mach_8"]
        total_temperature_5 = inputs["total_temperature_5"]

        partials["static_temperature_8", "total_temperature_5"] = np.diag(
            1.0 / (1.0 + (gamma_5 - 1.0) / 2 * mach_8 ** 2.0)
        )
        partials["static_temperature_8", "gamma_5"] = np.diag(
            -total_temperature_5
            / (1.0 + (gamma_5 - 1.0) / 2 * mach_8 ** 2.0) ** 2.0
            * mach_8 ** 2.0
            / 2.0
        )
        partials["static_temperature_8", "mach_8"] = np.diag(
            -total_temperature_5
            / (1.0 + (gamma_5 - 1.0) / 2 * mach_8 ** 2.0) ** 2.0
            * mach_8
            * (gamma_5 - 1.0)
        )


class Station8Velocity(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("gamma_5", shape=n, val=np.nan)
        self.add_input("mach_8", shape=n, val=np.nan)
        self.add_input("static_temperature_8", units="K", shape=n, val=np.nan)

        self.add_output("velocity_8", units="m/s", shape=n, val=1e2)

        self.declare_partials(of="velocity_8", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        r_g = 287.0  # Perfect gas constant

        gamma_5 = inputs["gamma_5"]
        mach_8 = inputs["mach_8"]
        static_temperature_8 = inputs["static_temperature_8"]

        outputs["velocity_8"] = mach_8 * np.sqrt(gamma_5 * r_g * static_temperature_8)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        r_g = 287.0  # Perfect gas constant

        gamma_5 = inputs["gamma_5"]
        mach_8 = inputs["mach_8"]
        static_temperature_8 = inputs["static_temperature_8"]

        partials["velocity_8", "mach_8"] = np.diag(np.sqrt(gamma_5 * r_g * static_temperature_8))
        partials["velocity_8", "gamma_5"] = np.diag(
            mach_8 / 2.0 * np.sqrt(r_g * static_temperature_8 / gamma_5)
        )
        partials["velocity_8", "static_temperature_8"] = np.diag(
            mach_8 / 2.0 * np.sqrt(r_g / static_temperature_8 * gamma_5)
        )

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

from .turboshaft_off_design_fuel import Turboshaft
from .propeller_thrust import PropellerMaxThrust


class TurboshaftMaxThrustPowerLimit(Turboshaft):
    def setup(self):

        n = self.options["number_of_points"]

        self.add_subsystem(
            "propeller_max_thrust",
            PropellerMaxThrust(number_of_points=n),
            promotes=["*"],
        )

        self.add_subsystem(
            "distance_to_limit_power",
            DistanceToLimitPowerLimit(number_of_points=n),
            promotes=["*"],
        )

        super().setup()


class DistanceToLimitPowerLimit(om.ImplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("shaft_power", units="kW", shape=n, val=np.nan)
        self.add_input("shaft_power_limit", units="kW", shape=n, val=np.nan)

        self.add_output("required_thrust", units="kN", val=np.full(n, 5.0))

        self.declare_partials(of="*", wrt="*", method="exact")

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        shaft_power = inputs["shaft_power"]
        shaft_power_limit = inputs["shaft_power_limit"]

        residuals["required_thrust"] = shaft_power / shaft_power_limit - 1.0
        # print("Constraints", total_temperature_45, opr, shaft_power)

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):

        shaft_power = inputs["shaft_power"]
        shaft_power_limit = inputs["shaft_power_limit"]

        jacobian["required_thrust", "shaft_power"] = np.diag(1.0 / shaft_power_limit)
        jacobian["required_thrust", "shaft_power_limit"] = np.diag(
            -shaft_power / shaft_power_limit ** 2.0
        )

        jacobian["required_thrust", "required_thrust"] = np.diag(np.zeros_like(shaft_power))


####################################################################################################


class TurboshaftMaxThrustOPRLimit(Turboshaft):
    def setup(self):

        n = self.options["number_of_points"]

        self.add_subsystem(
            "propeller_max_thrust",
            PropellerMaxThrust(number_of_points=n),
            promotes=["*"],
        )

        self.add_subsystem(
            "distance_to_limit_opr_limit",
            DistanceToLimitOPRLimit(number_of_points=n),
            promotes=["*"],
        )

        super().setup()


class DistanceToLimitOPRLimit(om.ImplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("opr", shape=n, val=np.nan)
        self.add_input("opr_limit", shape=n, val=np.nan)

        self.add_output("required_thrust", units="kN", val=np.full(n, 5.0))

        self.declare_partials(of="*", wrt="*", method="exact")

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        opr = inputs["opr"]
        opr_limit = inputs["opr_limit"]

        residuals["required_thrust"] = opr / opr_limit - 1.0

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):

        opr = inputs["opr"]
        opr_limit = inputs["opr_limit"]

        jacobian["required_thrust", "opr"] = np.diag(1.0 / opr_limit)
        jacobian["required_thrust", "opr_limit"] = np.diag(-opr / opr_limit ** 2.0)

        jacobian["required_thrust", "required_thrust"] = np.diag(np.zeros_like(opr))


####################################################################################################


class TurboshaftMaxThrustITTLimit(Turboshaft):
    def setup(self):

        n = self.options["number_of_points"]

        self.add_subsystem(
            "propeller_max_thrust",
            PropellerMaxThrust(number_of_points=n),
            promotes=["*"],
        )

        self.add_subsystem(
            "distance_to_limit_itt_limit",
            DistanceToLimitITTLimit(number_of_points=n),
            promotes=["*"],
        )

        super().setup()


class DistanceToLimitITTLimit(om.ImplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("total_temperature_45", units="degK", shape=n, val=np.nan)
        self.add_input("itt_limit", units="degK", shape=n, val=np.nan)

        self.add_output("required_thrust", units="kN", val=np.full(n, 5.0))

        self.declare_partials(of="*", wrt="*", method="exact")

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        total_temperature_45 = inputs["total_temperature_45"]
        itt_limit = inputs["itt_limit"]

        residuals["required_thrust"] = total_temperature_45 / itt_limit - 1.0

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):

        total_temperature_45 = inputs["total_temperature_45"]
        itt_limit = inputs["itt_limit"]

        jacobian["required_thrust", "total_temperature_45"] = np.diag(1.0 / itt_limit)
        jacobian["required_thrust", "itt_limit"] = np.diag(-total_temperature_45 / itt_limit ** 2.0)

        jacobian["required_thrust", "required_thrust"] = np.diag(
            np.zeros_like(total_temperature_45)
        )


####################################################################################################


class TurboshaftMaxThrustPropellerThrustLimit(Turboshaft):
    def setup(self):

        n = self.options["number_of_points"]

        self.add_subsystem(
            "propeller_max_thrust",
            PropellerMaxThrust(number_of_points=n),
            promotes=["*"],
        )

        self.add_subsystem(
            "distance_to_limit_propeller_thrust_limit",
            DistanceToLimitPropellerThrustLimit(number_of_points=n),
            promotes=["*"],
        )

        super().setup()


class DistanceToLimitPropellerThrustLimit(om.ImplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("propeller_thrust", units="N", shape=n, val=np.nan)
        self.add_input("propeller_max_thrust", units="N", shape=n, val=np.nan)

        self.add_output("required_thrust", units="kN", val=np.full(n, 5.0))

        self.declare_partials(
            of="required_thrust", wrt=["propeller_max_thrust", "propeller_thrust"], method="exact"
        )

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        propeller_thrust = inputs["propeller_thrust"]
        propeller_max_thrust = inputs["propeller_max_thrust"]

        residuals["required_thrust"] = propeller_thrust / propeller_max_thrust - 1.0

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):

        propeller_thrust = inputs["propeller_thrust"]
        propeller_max_thrust = inputs["propeller_max_thrust"]

        jacobian["required_thrust", "propeller_thrust"] = np.diag(1.0 / propeller_max_thrust)
        jacobian["required_thrust", "propeller_max_thrust"] = np.diag(
            -propeller_thrust / propeller_max_thrust ** 2.0
        )

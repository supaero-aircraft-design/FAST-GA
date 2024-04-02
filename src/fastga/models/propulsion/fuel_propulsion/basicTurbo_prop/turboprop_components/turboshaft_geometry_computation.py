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


import openmdao.api as om

from ..resources.read_resources import read_air_coeff, read_pressurization_coeff

from .station_0 import (
    Station0,
    Station0Static,
)
from .pressurization_air_flow import (
    PressurizationAirFlow,
)
from .station_02 import Station02
from .mass_flow import MassFlow
from .opr import OverallPressureRatioDesignPoint
from .station_225 import Station225DesignPoint
from .station_253 import (
    Station253Temperature,
    Station253PressureDesignPoint,
)
from .station_341 import Station341Pressure
from .station_441 import Station441PressureDesignPoint
from .thermodynamic_equilibrium_design import (
    ThermodynamicEquilibriumDesignPoint,
)
from .alpha import AlphaRatio
from .area_combustion_chamber import A41
from .area_inter_turbine import A45
from .area_exhaust import A81, A82, A8

from .air_coefficient import AirCoefficientReader


class DesignPointCalculation(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.temperature = None
        self.cv = None
        self.cp = None
        self.gamma = None

        self.flight_altitude = None
        self.cabin_altitude = None

    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        if self.temperature is None:
            self.temperature, self.cv, self.cp, self.gamma = read_air_coeff()

        if self.flight_altitude is None:
            self.flight_altitude, self.cabin_altitude = read_pressurization_coeff()

        n = self.options["number_of_points"]

        self.add_subsystem(
            "station_0_static",
            Station0Static(number_of_points=n, design_point=True),
            promotes=["*"],
        )
        self.add_subsystem(
            "pressurization_air_flow",
            PressurizationAirFlow(
                number_of_points=n,
                altitude_list=self.flight_altitude,
                cabin_altitude_list=self.cabin_altitude,
                design_point=True,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "station_0_total",
            Station0(number_of_points=n, design_point=True),
            promotes=["*"],
        )
        self.add_subsystem("station_02", Station02(number_of_points=n), promotes=["*"])

        self.add_subsystem("mass_flow", MassFlow(number_of_points=n), promotes=["*"])

        self.add_subsystem(
            "air_coeff_2",
            AirCoefficientReader(
                number_of_points=n,
                temperature_list=self.temperature,
                cv_list=self.cv,
                cp_list=self.cp,
                gamma_list=self.gamma,
            ),
            promotes=[],
        )

        self.connect("total_temperature_2", "air_coeff_2.total_temperature")

        self.connect("air_coeff_2.cp", "cp_2")
        self.connect("air_coeff_2.gamma", "gamma_2")

        self.add_subsystem(
            "opr_design_point",
            OverallPressureRatioDesignPoint(number_of_points=n),
            promotes=["*"],
        )
        self.add_subsystem(
            "station_225",
            Station225DesignPoint(number_of_points=n),
            promotes=["*"],
        )
        self.add_subsystem(
            "station_253_pressure",
            Station253PressureDesignPoint(number_of_points=n),
            promotes=["*"],
        )

        self.add_subsystem(
            "air_coeff_25",
            AirCoefficientReader(
                number_of_points=n,
                temperature_list=self.temperature,
                cv_list=self.cv,
                cp_list=self.cp,
                gamma_list=self.gamma,
            ),
            promotes=[],
        )

        self.connect("total_temperature_25", "air_coeff_25.total_temperature")

        self.connect("air_coeff_25.cp", "cp_25")
        self.connect("air_coeff_25.gamma", "gamma_25")

        self.add_subsystem(
            "station_253_temperature",
            Station253Temperature(number_of_points=n),
            promotes=["*"],
        )
        self.add_subsystem(
            "station_341_pressure",
            Station341Pressure(number_of_points=n),
            promotes=["*"],
        )
        self.add_subsystem(
            "station_441_pressure",
            Station441PressureDesignPoint(number_of_points=n),
            promotes=["*"],
        )

        self.add_subsystem(
            "air_coeff_3",
            AirCoefficientReader(
                number_of_points=n,
                temperature_list=self.temperature,
                cv_list=self.cv,
                cp_list=self.cp,
                gamma_list=self.gamma,
            ),
            promotes=[],
        )

        self.connect("total_temperature_3", "air_coeff_3.total_temperature")

        self.connect("air_coeff_3.cp", "cp_3")

        self.add_subsystem(
            "air_coeff_4",
            AirCoefficientReader(
                number_of_points=n,
                temperature_list=self.temperature,
                cv_list=self.cv,
                cp_list=self.cp,
                gamma_list=self.gamma,
            ),
            promotes=[],
        )

        self.connect("total_temperature_4", "air_coeff_4.total_temperature")

        self.connect("air_coeff_4.cp", "cp_4")

        self.add_subsystem(
            "air_coeff_41",
            AirCoefficientReader(
                number_of_points=n,
                temperature_list=self.temperature,
                cv_list=self.cv,
                cp_list=self.cp,
                gamma_list=self.gamma,
                design_point=True,
            ),
            promotes=["data:*"],
        )

        self.connect("air_coeff_41.cp", "cp_41")
        self.connect("air_coeff_41.gamma", "gamma_41")

        self.add_subsystem(
            "air_coeff_45",
            AirCoefficientReader(
                number_of_points=n,
                temperature_list=self.temperature,
                cv_list=self.cv,
                cp_list=self.cp,
                gamma_list=self.gamma,
            ),
            promotes=[],
        )

        self.connect("total_temperature_45", "air_coeff_45.total_temperature")

        self.connect("air_coeff_45.cp", "cp_45")
        self.connect("air_coeff_45.gamma", "gamma_45")

        self.add_subsystem(
            "air_coeff_5",
            AirCoefficientReader(
                number_of_points=n,
                temperature_list=self.temperature,
                cv_list=self.cv,
                cp_list=self.cp,
                gamma_list=self.gamma,
            ),
            promotes=[],
        )

        self.connect("total_temperature_5", "air_coeff_5.total_temperature")

        self.connect("air_coeff_5.cp", "cp_5")
        self.connect("air_coeff_5.gamma", "gamma_5")

        self.add_subsystem(
            "thermodynamic_equilibrium_design_point",
            ThermodynamicEquilibriumDesignPoint(number_of_points=n),
            promotes=["*"],
        )

        self.add_subsystem(
            "alpha_ratio",
            AlphaRatio(number_of_points=n),
            promotes=["*"],
        )

        self.add_subsystem(
            "area_combustion_chamber",
            A41(number_of_points=n),
            promotes=["*"],
        )
        self.add_subsystem(
            "area_inter_turbine",
            A45(number_of_points=n),
            promotes=["*"],
        )

        self.add_subsystem(
            "area_exhaust_part_1",
            A81(number_of_points=n),
            promotes=["*"],
        )
        self.add_subsystem(
            "area_exhaust_part_2",
            A82(number_of_points=n),
            promotes=["*"],
        )
        self.add_subsystem(
            "area_exhaust",
            A8(number_of_points=n),
            promotes=["*"],
        )

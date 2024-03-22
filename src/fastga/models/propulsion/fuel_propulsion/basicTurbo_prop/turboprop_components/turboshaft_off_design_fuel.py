import openmdao.api as om

from .air_coefficient import AirCoefficientReader
from .exhaust_thrust import ExhaustThrust
from .balance_power import BalancePower

from .station_0 import Station0, Station0Static
from .pressurization_air_flow import PressurizationAirFlow
from .station_02 import Station02
from .station_4145 import (
    Station4145Temperature,
    Station4145Pressure,
)
from .mass_flow import MassFlow
from .station_225 import Station225Pressure
from .station_253 import (
    Station253Pressure,
    Station253Temperature,
)
from .opr import OverallPressureRatio
from .station_441 import Station441Temperature
from .station_341 import Station341Pressure
from .thermodynamic_equilibrium import (
    ThermodynamicEquilibrium,
)
from .station_8 import (
    Station8Mach,
    Station8Temperature,
    Station8Velocity,
)
from .station_58 import Station58Pressure
from .exhaust_equilibrium import ExhaustEquilibrium
from .shaft_power import ShaftPower
from .propeller_thrust import (
    PropellerThrustRequired,
    ShaftPowerRequired,
)

from ..resources.read_resources import read_air_coeff, read_pressurization_coeff


class Turboshaft(om.Group):
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

        self.add_subsystem("station_0_static", Station0Static(number_of_points=n), promotes=["*"])
        self.add_subsystem(
            "pressurization_air_flow",
            PressurizationAirFlow(
                number_of_points=n,
                altitude_list=self.flight_altitude,
                cabin_altitude_list=self.cabin_altitude,
            ),
            promotes=["*"],
        )
        self.add_subsystem("station_0_total", Station0(number_of_points=n), promotes=["*"])
        self.add_subsystem("station_02", Station02(number_of_points=n), promotes=["*"])

        self.add_subsystem("mass_flow", MassFlow(number_of_points=n), promotes=["*"])

        self.add_subsystem(
            "station_4145_temperature",
            Station4145Temperature(number_of_points=n),
            promotes=["*"],
        )

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
        self.add_subsystem(
            "air_coeff_41",
            AirCoefficientReader(
                number_of_points=n,
                temperature_list=self.temperature,
                cv_list=self.cv,
                cp_list=self.cp,
                gamma_list=self.gamma,
            ),
            promotes=[],
        )
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

        self.connect("total_temperature_2", "air_coeff_2.total_temperature")
        self.connect("total_temperature_25", "air_coeff_25.total_temperature")
        self.connect("total_temperature_41", "air_coeff_41.total_temperature")
        self.connect("total_temperature_45", "air_coeff_45.total_temperature")

        self.connect("air_coeff_2.cp", "cp_2")
        self.connect("air_coeff_2.gamma", "gamma_2")
        self.connect("air_coeff_25.cp", "cp_25")
        self.connect("air_coeff_25.gamma", "gamma_25")
        self.connect("air_coeff_41.cp", "cp_41")
        self.connect("air_coeff_41.gamma", "gamma_41")
        self.connect("air_coeff_45.cp", "cp_45")
        self.connect("air_coeff_45.gamma", "gamma_45")

        self.add_subsystem(
            "station_225_pressure",
            Station225Pressure(number_of_points=n),
            promotes=["*"],
        )
        self.add_subsystem(
            "station_253_pressure",
            Station253Pressure(number_of_points=n),
            promotes=["*"],
        )
        self.add_subsystem(
            "station_253_temperature",
            Station253Temperature(number_of_points=n),
            promotes=["*"],
        )
        self.add_subsystem("opr", OverallPressureRatio(number_of_points=n), promotes=["*"])

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
            "station_441_temperature",
            Station441Temperature(number_of_points=n),
            promotes=["*"],
        )

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
            "station_341_temperature",
            Station341Pressure(number_of_points=n),
            promotes=["*"],
        )

        self.add_subsystem(
            "thermodynamic_equilibrium",
            ThermodynamicEquilibrium(number_of_points=n),
            promotes=["*"],
        )

        self.add_subsystem(
            "station_4145_pressure",
            Station4145Pressure(number_of_points=n),
            promotes=["*"],
        )

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
        self.connect("air_coeff_5.gamma", "gamma_5")
        self.connect("air_coeff_5.cp", "cp_5")

        self.add_subsystem("station_8_mach", Station8Mach(number_of_points=n), promotes=["*"])
        self.add_subsystem("station_58", Station58Pressure(number_of_points=n), promotes=["*"])

        self.add_subsystem(
            "exhaust_equilibrium",
            ExhaustEquilibrium(number_of_points=n),
            promotes=["*"],
        )

        self.add_subsystem(
            "shaft_power_computation",
            ShaftPower(number_of_points=n),
            promotes=["*"],
        )

        self.add_subsystem(
            "station_8_temperature",
            Station8Temperature(number_of_points=n),
            promotes=["*"],
        )

        self.add_subsystem(
            "station_8_velocity",
            Station8Velocity(number_of_points=n),
            promotes=["*"],
        )

        self.add_subsystem(
            "exhaust_thrust",
            ExhaustThrust(number_of_points=n),
            promotes=["*"],
        )

        self.add_subsystem(
            "required_prop_thrust",
            PropellerThrustRequired(number_of_points=n),
            promotes=["*"],
        )

        self.add_subsystem(
            "shaft_power_required",
            ShaftPowerRequired(number_of_points=n),
            promotes=["*"],
        )

        self.add_subsystem(
            "balance_power",
            BalancePower(number_of_points=n),
            promotes=["*"],
        )

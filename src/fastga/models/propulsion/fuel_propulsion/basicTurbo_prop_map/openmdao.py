"""OpenMDAO wrapping of basic IC engine."""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
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
from openmdao.core.component import Component

import fastoad.api as oad

from fastga.models.propulsion.fuel_propulsion.basicTurbo_prop_map.basicTP_engine_mapped import (
    BasicTPEngineMapped,
)

from fastga.models.propulsion.propulsion import IPropulsion, BaseOMPropulsionComponent
from fastga.models.propulsion.fuel_propulsion.base import FuelEngineSet
from fastga.models.aerodynamics.external.propeller_code.compute_propeller_aero import (
    THRUST_PTS_NB,
    SPEED_PTS_NB,
)
from fastga.models.propulsion.fuel_propulsion.basicTurbo_prop_map.basicTP_engine_constructor import (
    THRUST_PTS_NB_TURBOPROP,
    MACH_PTS_NB_TURBOPROP,
)


@oad.RegisterPropulsion("fastga.wrapper.propulsion.basicTurbopropMapped")
class OMBasicTurbopropMapWrapper(oad.IOMPropulsionWrapper):
    """
    Wrapper class for basic Turboprop model using precalculated maps.
    It is made to allow a direct call to :class:`~.basicIC_engine.BasicTPEngineMapped` in an
    OpenMDAO component.
    Example of usage of this class::
        import openmdao.api as om
        class MyComponent(om.ExplicitComponent):
            def initialize():
                self._engine_wrapper = OMBasicTurbopropMapWrapper()
            def setup():
                # Adds OpenMDAO variables that define the engine
                self._engine_wrapper.setup(self)
                # Do the normal setup
                self.add_input("my_input")
                [finish the setup...]
            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                [do something]
                # Get the engine instance, with parameters defined from OpenMDAO inputs
                engine = self._engine_wrapper.get_model(inputs)
                # Run the engine model. This is a pure Python call. You have to define
                # its inputs before, and to use its outputs according to your needs
                sfc, thrust_rate, thrust = engine.compute_flight_points(
                    mach,
                    altitude,
                    engine_setting,
                    thrust_is_regulated,
                    thrust_rate,
                    thrust
                    )
                [do something else]
        )
    """

    def setup(self, component: Component):
        component.add_input("data:propulsion:turboprop:design_point:power", np.nan, units="kW")
        component.add_input(
            "data:propulsion:turboprop:design_point:turbine_entry_temperature", np.nan, units="K"
        )
        component.add_input("data:propulsion:turboprop:design_point:OPR", np.nan)
        component.add_input("data:propulsion:turboprop:design_point:altitude", np.nan, units="m")
        component.add_input("data:propulsion:turboprop:design_point:mach", np.nan)
        component.add_input("data:propulsion:turboprop:off_design:bleed_usage", np.nan)
        component.add_input("data:propulsion:turboprop:off_design:itt_limit", np.nan, units="K")
        component.add_input("data:propulsion:turboprop:off_design:power_limit", np.nan, units="kW")
        component.add_input("data:propulsion:turboprop:off_design:opr_limit", np.nan)
        component.add_input("data:aerodynamics:propeller:cruise_level:altitude", np.nan, units="m")
        component.add_input("data:geometry:propulsion:engine:layout", np.nan)
        component.add_input("data:geometry:propulsion:engine:count", val=np.nan)
        component.add_input(
            "data:aerodynamics:propeller:sea_level:speed",
            np.full(SPEED_PTS_NB, np.nan),
            units="m/s",
        )
        component.add_input(
            "data:aerodynamics:propeller:sea_level:thrust",
            np.full(THRUST_PTS_NB, np.nan),
            units="N",
        )
        component.add_input(
            "data:aerodynamics:propeller:sea_level:thrust_limit",
            np.full(SPEED_PTS_NB, np.nan),
            units="N",
        )
        component.add_input(
            "data:aerodynamics:propeller:sea_level:efficiency",
            np.full((SPEED_PTS_NB, THRUST_PTS_NB), np.nan),
        )
        component.add_input(
            "data:aerodynamics:propeller:cruise_level:speed",
            np.full(SPEED_PTS_NB, np.nan),
            units="m/s",
        )
        component.add_input(
            "data:aerodynamics:propeller:cruise_level:thrust",
            np.full(THRUST_PTS_NB, np.nan),
            units="N",
        )
        component.add_input(
            "data:aerodynamics:propeller:cruise_level:thrust_limit",
            np.full(SPEED_PTS_NB, np.nan),
            units="N",
        )
        component.add_input(
            "data:aerodynamics:propeller:cruise_level:efficiency",
            np.full((SPEED_PTS_NB, THRUST_PTS_NB), np.nan),
        )
        component.add_input(
            "data:aerodynamics:propeller:installation_effect:effective_efficiency:low_speed",
            val=1.0,
        )
        component.add_input(
            "data:aerodynamics:propeller:installation_effect:effective_efficiency:cruise",
            val=1.0,
        )
        component.add_input(
            "data:aerodynamics:propeller:installation_effect:effective_advance_ratio",
            val=1.0,
        )

        component.add_input(
            "settings:propulsion:turboprop:efficiency:first_compressor_stage", val=0.85
        )
        component.add_input(
            "settings:propulsion:turboprop:efficiency:second_compressor_stage", val=0.86
        )
        component.add_input(
            "settings:propulsion:turboprop:efficiency:high_pressure_turbine", val=0.86
        )
        component.add_input("settings:propulsion:turboprop:efficiency:power_turbine", val=0.86)
        component.add_input(
            "settings:propulsion:turboprop:efficiency:combustion", val=43.260e6 * 0.95, units="J/kg"
        )
        component.add_input("settings:propulsion:turboprop:efficiency:high_pressure_axe", val=0.98)
        component.add_input("settings:propulsion:turboprop:pressure_loss:inlet", val=0.8)
        component.add_input(
            "settings:propulsion:turboprop:pressure_loss:combustion_chamber", val=0.95
        )
        component.add_input("settings:propulsion:turboprop:bleed:turbine_cooling", val=0.05)
        component.add_input(
            "settings:propulsion:turboprop:electric_power_offtake", val=50 * 745.7, units="W"
        )
        component.add_input("settings:propulsion:turboprop:efficiency:gearbox", val=0.98)
        component.add_input("settings:propulsion:turboprop:bleed:inter_compressor", val=0.04)
        component.add_input("settings:propulsion:turboprop:design_point:mach_exhaust", val=0.4)
        component.add_input(
            "settings:propulsion:turboprop:design_point:first_stage_pressure_ratio", val=0.25
        )

        component.add_input(
            "data:propulsion:turboprop:sea_level:mach",
            shape=MACH_PTS_NB_TURBOPROP,
        )
        component.add_input(
            "data:propulsion:turboprop:sea_level:thrust", shape=THRUST_PTS_NB_TURBOPROP, units="N"
        )
        component.add_input(
            "data:propulsion:turboprop:sea_level:thrust_limit",
            shape=MACH_PTS_NB_TURBOPROP,
            units="N",
        )
        component.add_input(
            "data:propulsion:turboprop:sea_level:sfc",
            shape=(MACH_PTS_NB_TURBOPROP, THRUST_PTS_NB_TURBOPROP),
            units="kg/s/N",
        )

        component.add_input(
            "data:propulsion:turboprop:cruise_level:mach",
            shape=MACH_PTS_NB_TURBOPROP,
        )
        component.add_input(
            "data:propulsion:turboprop:cruise_level:thrust",
            shape=THRUST_PTS_NB_TURBOPROP,
            units="N",
        )
        component.add_input(
            "data:propulsion:turboprop:cruise_level:thrust_limit",
            shape=MACH_PTS_NB_TURBOPROP,
            units="N",
        )
        component.add_input(
            "data:propulsion:turboprop:cruise_level:sfc",
            shape=(MACH_PTS_NB_TURBOPROP, THRUST_PTS_NB_TURBOPROP),
            units="kg/s/N",
        )

        component.add_input("data:propulsion:turboprop:intermediate_level:altitude", units="m")
        component.add_input(
            "data:propulsion:turboprop:intermediate_level:mach",
            shape=MACH_PTS_NB_TURBOPROP,
        )
        component.add_input(
            "data:propulsion:turboprop:intermediate_level:thrust",
            shape=THRUST_PTS_NB_TURBOPROP,
            units="N",
        )
        component.add_input(
            "data:propulsion:turboprop:intermediate_level:thrust_limit",
            shape=MACH_PTS_NB_TURBOPROP,
            units="N",
        )
        component.add_input(
            "data:propulsion:turboprop:intermediate_level:sfc",
            shape=(MACH_PTS_NB_TURBOPROP, THRUST_PTS_NB_TURBOPROP),
            units="kg/s/N",
        )
        # TODO: add effective efficiency

    @staticmethod
    def get_model(inputs) -> IPropulsion:
        """
        :param inputs: input parameters that define the engine
        :return: an :class:`BasicTPEngine` instance
        """
        engine_params = {
            "power_design": inputs["data:propulsion:turboprop:design_point:power"],
            "t_41t_design": inputs[
                "data:propulsion:turboprop:design_point:turbine_entry_temperature"
            ],
            "opr_design": inputs["data:propulsion:turboprop:design_point:OPR"],
            "cruise_altitude_propeller": inputs[
                "data:aerodynamics:propeller:cruise_level:altitude"
            ],
            "design_altitude": inputs["data:propulsion:turboprop:design_point:altitude"],
            "design_mach": inputs["data:propulsion:turboprop:design_point:mach"],
            "prop_layout": inputs["data:geometry:propulsion:engine:layout"],
            "bleed_control": inputs["data:propulsion:turboprop:off_design:bleed_usage"],
            "itt_limit": inputs["data:propulsion:turboprop:off_design:itt_limit"],
            "power_limit": inputs["data:propulsion:turboprop:off_design:power_limit"],
            "opr_limit": inputs["data:propulsion:turboprop:off_design:opr_limit"],
            "speed_SL": inputs["data:aerodynamics:propeller:sea_level:speed"],
            "thrust_SL": inputs["data:aerodynamics:propeller:sea_level:thrust"],
            "thrust_limit_SL": inputs["data:aerodynamics:propeller:sea_level:thrust_limit"],
            "efficiency_SL": inputs["data:aerodynamics:propeller:sea_level:efficiency"],
            "speed_CL": inputs["data:aerodynamics:propeller:cruise_level:speed"],
            "thrust_CL": inputs["data:aerodynamics:propeller:cruise_level:thrust"],
            "thrust_limit_CL": inputs["data:aerodynamics:propeller:cruise_level:thrust_limit"],
            "efficiency_CL": inputs["data:aerodynamics:propeller:cruise_level:efficiency"],
            "effective_J": inputs[
                "data:aerodynamics:propeller:installation_effect:effective_advance_ratio"
            ],
            "effective_efficiency_ls": inputs[
                "data:aerodynamics:propeller:installation_effect:effective_efficiency:low_speed"
            ],
            "effective_efficiency_cruise": inputs[
                "data:aerodynamics:propeller:installation_effect:effective_efficiency:cruise"
            ],
            "turbo_mach_SL": inputs["data:propulsion:turboprop:sea_level:mach"],
            "turbo_thrust_SL": inputs["data:propulsion:turboprop:sea_level:thrust"],
            "turbo_thrust_max_SL": inputs["data:propulsion:turboprop:sea_level:thrust_limit"],
            "turbo_sfc_SL": inputs["data:propulsion:turboprop:sea_level:sfc"],
            "turbo_mach_CL": inputs["data:propulsion:turboprop:cruise_level:mach"],
            "turbo_thrust_CL": inputs["data:propulsion:turboprop:cruise_level:thrust"],
            "turbo_thrust_max_CL": inputs["data:propulsion:turboprop:cruise_level:thrust_limit"],
            "turbo_sfc_CL": inputs["data:propulsion:turboprop:cruise_level:sfc"],
            "turbo_mach_IL": inputs["data:propulsion:turboprop:intermediate_level:mach"],
            "turbo_thrust_IL": inputs["data:propulsion:turboprop:intermediate_level:thrust"],
            "turbo_thrust_max_IL": inputs[
                "data:propulsion:turboprop:intermediate_level:thrust_limit"
            ],
            "turbo_sfc_IL": inputs["data:propulsion:turboprop:intermediate_level:sfc"],
            "level_IL": inputs["data:propulsion:turboprop:intermediate_level:altitude"],
            "eta_225": inputs["settings:propulsion:turboprop:efficiency:first_compressor_stage"],
            "eta_253": inputs["settings:propulsion:turboprop:efficiency:second_compressor_stage"],
            "eta_445": inputs["settings:propulsion:turboprop:efficiency:high_pressure_turbine"],
            "eta_455": inputs["settings:propulsion:turboprop:efficiency:power_turbine"],
            "eta_q": inputs["settings:propulsion:turboprop:efficiency:combustion"],
            "eta_axe": inputs["settings:propulsion:turboprop:efficiency:high_pressure_axe"],
            "pi_02": inputs["settings:propulsion:turboprop:pressure_loss:inlet"],
            "pi_cc": inputs["settings:propulsion:turboprop:pressure_loss:combustion_chamber"],
            "cooling_ratio": inputs["settings:propulsion:turboprop:bleed:turbine_cooling"],
            "hp_shaft_power_out": inputs["settings:propulsion:turboprop:electric_power_offtake"],
            "gearbox_efficiency": inputs["settings:propulsion:turboprop:efficiency:gearbox"],
            "inter_compressor_bleed": inputs[
                "settings:propulsion:turboprop:bleed:inter_compressor"
            ],
            "exhaust_mach_design": inputs[
                "settings:propulsion:turboprop:design_point:mach_exhaust"
            ],
            "pr_1_ratio_design": inputs[
                "settings:propulsion:turboprop:design_point:first_stage_pressure_ratio"
            ],
        }

        return FuelEngineSet(
            BasicTPEngineMapped(**engine_params), inputs["data:geometry:propulsion:engine:count"]
        )


@oad.ValidityDomainChecker(
    {
        "data:propulsion:turboprop:max_power": (
            180,
            10000,
        ),  # power range validity for design point
        "data:propulsion:turboprop:design_point:turbine_entry_temperature": (
            500.0,
            2000.0,
        ),  # turbine entry
        # temperature validity for design point
        "data:propulsion:turboprop:design_point:OPR": (1.1, 42.0),  # opr validity for design point
        "data:geometry:propulsion:engine:layout": [1.0, 3.0],  # propulsion position (3.0=Nose,
        # 1.0=Wing)
        "data:propulsion:turboprop:off_design:bleed_usage": [
            0.0,
            1.0,
        ],  # bleed usage at the design point (0.0="low",
        # 1.0="high")
        "data:propulsion:turboprop:off_design:itt_limit": (500.0, 1000.0),  # turbine entry
        # temperature validity for off design
        "data:propulsion:turboprop:off_design:power_limit": (
            180,
            10000,
        ),  # power range validity for off design
        "data:propulsion:turboprop:off_design:opr_limit": (
            1.1,
            42.0,
        ),  # opr range validity for off design
    }
)
class OMBasicTPEngineMappedComponent(BaseOMPropulsionComponent):
    """
    Parametric engine model as OpenMDAO component
    See :class:`BasicICEngine` for more information.
    """

    def setup(self):
        super().setup()
        self.get_wrapper().setup(self)

    @staticmethod
    def get_wrapper() -> OMBasicTurbopropMapWrapper:
        return OMBasicTurbopropMapWrapper()

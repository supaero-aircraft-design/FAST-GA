"""OpenMDAO wrapping of basic IC engine."""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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

from fastoad.model_base.propulsion import IOMPropulsionWrapper
from fastoad.module_management.service_registry import RegisterPropulsion
from fastoad.openmdao.validity_checker import ValidityDomainChecker

from .basicHE_engine import BasicHEEngine

from fastga.models.propulsion.propulsion import IPropulsion, BaseOMPropulsionComponent
from fastga.models.aerodynamics.components.compute_propeller_aero import THRUST_PTS_NB, SPEED_PTS_NB


@RegisterPropulsion("fastga.wrapper.propulsion.basicHE_engine")
class OMBasicHEEngineWrapper(IOMPropulsionWrapper):
    """
    Wrapper class for basic HE engine model.
    It is made to allow a direct call to :class:`~.basicHE_engine.BasicHEEngine` in an OpenMDAO
    component.
    Example of usage of this class::
        import openmdao.api as om
        class MyComponent(om.ExplicitComponent):
            def initialize():
                self._engine_wrapper = OMRubberEngineWrapper()
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
        component.add_input("data:propulsion:HE_engine:max_power", np.nan, units="W")
        component.add_input("data:TLAR:v_cruise", np.nan, units="m/s")
        component.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")
        component.add_input("data:geometry:propulsion:layout", np.nan)
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
        component.add_input("data:propulsion:hybrid_powertrain:motor:speed", np.nan, units="rpm")
        component.add_input("data:propulsion:hybrid_powertrain:motor:nominal_torque", np.nan, units="N*m")
        component.add_input("data:propulsion:hybrid_powertrain:motor:max_torque", np.nan, units="N*m")
        component.add_input("data:propulsion:hybrid_powertrain:power_electronics:n_conv", np.nan)
        component.add_input("data:propulsion:hybrid_powertrain:fuel_cell:design_power", np.nan, units='W')
        component.add_input("data:propulsion:hybrid_powertrain:fuel_cell:hyd_mass_flow", np.nan, units='kg/s')
        component.add_input("data:propulsion:hybrid_powertrain:power_electronics:pe_specific_power", np.nan, units='W/kg')
        component.add_input("data:propulsion:hybrid_powertrain:cable:lsw", np.nan, units="kg/m")
        component.add_input("data:geometry:hybrid_powertrain:cables:length", np.nan, units="m")
        component.add_input("data:geometry:propeller:blades_number", np.nan, units=None)
        component.add_input("data:geometry:propeller:diameter", np.nan, units="m")
        component.add_input("data:geometry:propeller:prop_number", np.nan, units=None)
        component.add_input("settings:weight:hybrid_powertrain:prop_reduction_factor", np.nan, units=None)


    @staticmethod
    def get_model(inputs) -> IPropulsion:
        """
        :param inputs: input parameters that define the engine
        :return: an :class:`BasicHEEngine` instance
        """
        engine_params = {
            "max_power": inputs["data:propulsion:HE_engine:max_power"],
            "cruise_altitude": inputs["data:mission:sizing:main_route:cruise:altitude"],
            "cruise_speed": inputs["data:TLAR:v_cruise"],
            "prop_layout": inputs["data:geometry:propulsion:layout"],
            "speed_SL": inputs["data:aerodynamics:propeller:sea_level:speed"],
            "thrust_SL": inputs["data:aerodynamics:propeller:sea_level:thrust"],
            "thrust_limit_SL": inputs["data:aerodynamics:propeller:sea_level:thrust_limit"],
            "efficiency_SL": inputs["data:aerodynamics:propeller:sea_level:efficiency"],
            "speed_CL": inputs["data:aerodynamics:propeller:cruise_level:speed"],
            "thrust_CL": inputs["data:aerodynamics:propeller:cruise_level:thrust"],
            "thrust_limit_CL": inputs["data:aerodynamics:propeller:cruise_level:thrust_limit"],
            "efficiency_CL": inputs["data:aerodynamics:propeller:cruise_level:efficiency"],
            "motor_speed": inputs["data:propulsion:hybrid_powertrain:motor:speed"],
            "nominal_torque": inputs["data:propulsion:hybrid_powertrain:motor:nominal_torque"],
            "max_torque": inputs["data:propulsion:hybrid_powertrain:motor:max_torque"],
            "eta_pe": inputs["data:propulsion:hybrid_powertrain:power_electronics:n_conv"],
            "fc_des_power": inputs["data:propulsion:hybrid_powertrain:fuel_cell:design_power"],
            "H2_mass_flow": inputs["data:propulsion:hybrid_powertrain:fuel_cell:hyd_mass_flow"],
            "pe_specific_power": inputs["data:propulsion:hybrid_powertrain:power_electronics:pe_specific_power"],
            "cables_lsw": inputs["data:propulsion:hybrid_powertrain:cable:lsw"],
            "cables_length": inputs["data:geometry:hybrid_powertrain:cables:length"],
            "nb_blades": inputs["data:geometry:propeller:blades_number"],
            "prop_diameter": inputs["data:geometry:propeller:diameter"],
            "nb_propellers": inputs["data:geometry:propeller:prop_number"],
            "prop_red_factor": inputs["settings:weight:hybrid_powertrain:prop_reduction_factor"]
        }

        return BasicHEEngine(**engine_params)


@ValidityDomainChecker(
    {
        "data:propulsion:HE_engine:max_power": (50000, 250000),  # power range validity
        "data:geometry:propulsion:layout": [1.0, 3.0],  # propulsion position (3.0=Nose, 1.0=Wing)
    }
)
class OMBasicHEEngineComponent(BaseOMPropulsionComponent):
    """
    Parametric engine model as OpenMDAO component
    See :class:`BasicHEEngine` for more information.
    """

    def setup(self):
        super().setup()
        self.get_wrapper().setup(self)

    @staticmethod
    def get_wrapper() -> OMBasicHEEngineWrapper:
        return OMBasicHEEngineWrapper()

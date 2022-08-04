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

from fastga.models.propulsion.propulsion import IPropulsion, BaseOMPropulsionComponent
from fastga.models.propulsion.fuel_propulsion.basicIC_engine.basicIC_engine import BasicICEngine
from fastga.models.propulsion.fuel_propulsion.base import FuelEngineSet
from fastga.models.aerodynamics.external.propeller_code.compute_propeller_aero import (
    THRUST_PTS_NB,
    SPEED_PTS_NB,
)


@oad.RegisterPropulsion("fastga.wrapper.propulsion.basicIC_engine")
class OMBasicICEngineWrapper(oad.IOMPropulsionWrapper):
    """
    Wrapper class of for basic IC engine model.
    It is made to allow a direct call to :class:`~.basicIC_engine.BasicICEngine` in an OpenMDAO
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
        component.add_input("data:propulsion:IC_engine:max_power", np.nan, units="W")
        component.add_input("data:propulsion:fuel_type", np.nan)
        component.add_input("data:propulsion:IC_engine:strokes_nb", np.nan)
        component.add_input("data:geometry:propulsion:engine:layout", np.nan)
        component.add_input(
            "settings:propulsion:IC_engine:k_factor_sfc",
            1.0,
            desc="k_factor that can be used to adjust the consumption on engine level to the "
            "aircraft level",
        )
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
            "data:aerodynamics:propeller:cruise_level:altitude", units="m", val=np.nan
        )
        component.add_input("data:geometry:propulsion:engine:count", val=np.nan)
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

    @staticmethod
    def get_model(inputs) -> IPropulsion:
        """
        :param inputs: input parameters that define the engine
        :return: an :class:`BasicICEngine` instance
        """
        engine_params = {
            "max_power": inputs["data:propulsion:IC_engine:max_power"],
            "cruise_altitude_propeller": inputs[
                "data:aerodynamics:propeller:cruise_level:altitude"
            ],
            "fuel_type": inputs["data:propulsion:fuel_type"],
            "strokes_nb": inputs["data:propulsion:IC_engine:strokes_nb"],
            "prop_layout": inputs["data:geometry:propulsion:engine:layout"],
            "k_factor_sfc": inputs["settings:propulsion:IC_engine:k_factor_sfc"],
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
        }

        return FuelEngineSet(
            BasicICEngine(**engine_params), inputs["data:geometry:propulsion:engine:count"]
        )


@oad.ValidityDomainChecker(
    {
        "data:propulsion:IC_engine:max_power": (50000, 250000),  # power range validity
        "data:propulsion:fuel_type": [1.0, 2.0],  # fuel list
        "data:propulsion:IC_engine:strokes_nb": [2.0, 4.0],  # architecture list
        "data:geometry:propulsion:engine:layout": [
            1.0,
            3.0,
        ],  # propulsion position (3.0=Nose, 1.0=Wing)
        "data:aerodynamics:propeller:installation_effect:effective_advance_ratio": (0.0, 1.0),
        "data:aerodynamics:propeller:installation_effect:effective_efficiency:low_speed": (
            0.0,
            1.0,
        ),
        "data:aerodynamics:propeller:installation_effect:effective_efficiency:cruise": (0.0, 1.0),
    }
)
class OMBasicICEngineComponent(BaseOMPropulsionComponent):
    """
    Parametric engine model as OpenMDAO component
    See :class:`BasicICEngine` for more information.
    """

    def setup(self):
        super().setup()
        self.get_wrapper().setup(self)

    @staticmethod
    def get_wrapper() -> OMBasicICEngineWrapper:
        return OMBasicICEngineWrapper()

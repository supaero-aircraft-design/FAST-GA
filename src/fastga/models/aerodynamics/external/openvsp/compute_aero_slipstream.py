"""
    Estimation of slipstream effects using OPENVSP.
"""
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
import openmdao.api as om
from stdatm import Atmosphere

import fastoad.api as oad

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
from fastoad.constants import EngineSetting

from .openvsp import OPENVSPSimpleGeometryDP, DEFAULT_WING_AIRFOIL
from ...components.compute_reynolds import ComputeUnitReynolds
from ...constants import SPAN_MESH_POINT, SUBMODEL_THRUST_POWER_SLIPSTREAM

oad.RegisterSubmodel.active_models[
    SUBMODEL_THRUST_POWER_SLIPSTREAM
] = "fastga.submodel.aerodynamics.wing.slipstream.thrust_power_computation.via_id"


class ComputeSlipstreamOpenvsp(om.Group):
    """
    Computes the impact of the slipstream effects on the lift repartition of the aircraft by
    computing the difference between two OpenVSP runs, one with slipstream effects, one without.
    This group is meant to be used on its own, not with any other slipstream computation,
    hence why there is a computation of the Reynolds number.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
        self.options.declare("propulsion_id", default="", types=str)
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare("openvsp_exe_path", default="", types=str, allow_none=True)
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare(
            "wing_airfoil_file", default=DEFAULT_WING_AIRFOIL, types=str, allow_none=True
        )

    def setup(self):
        self.add_subsystem(
            "comp_unit_reynolds_slipstream",
            ComputeUnitReynolds(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["*"],
        )
        self.add_subsystem(
            "aero_slipstream_openvsp_subgroup",
            ComputeSlipstreamOpenvspSubGroup(
                propulsion_id=self.options["propulsion_id"],
                result_folder_path=self.options["result_folder_path"],
                openvsp_exe_path=self.options["openvsp_exe_path"],
                airfoil_folder_path=self.options["airfoil_folder_path"],
                wing_airfoil_file=self.options["wing_airfoil_file"],
                low_speed_aero=self.options["low_speed_aero"],
            ),
            promotes=["data:*"],
        )


class ComputeSlipstreamOpenvspSubGroup(om.Group):
    """
    Computes the impact of the slipstream effects on the lift repartition of the aircraft by
    computing the difference between two OpenVSP runs, one with slipstream effects, one without.
    This group is meant called in the AerodynamicsLowSpeed and AerodynamicsHighSpeed if the
    compute_slipstream option is set to True.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
        self.options.declare("propulsion_id", default="", types=str)
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare("openvsp_exe_path", default="", types=str, allow_none=True)
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare(
            "wing_airfoil_file", default=DEFAULT_WING_AIRFOIL, types=str, allow_none=True
        )

    def setup(self):
        self.add_subsystem(
            "comp_flight_conditions",
            FlightConditionsForDPComputation(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["data:*"],
        )

        propulsion_option = {"propulsion_id": self.options["propulsion_id"]}
        self.add_subsystem(
            "comp_thrust_power",
            oad.RegisterSubmodel.get_submodel(
                SUBMODEL_THRUST_POWER_SLIPSTREAM, options=propulsion_option
            ),
            promotes=["data:*"],
        )
        self.connect("comp_flight_conditions.mach", "comp_thrust_power.mach")
        self.connect("comp_flight_conditions.altitude", "comp_thrust_power.altitude")

        self.connect("comp_thrust_power.thrust", "aero_slipstream_openvsp.thrust")
        self.connect("comp_thrust_power.shaft_power", "aero_slipstream_openvsp.shaft_power")

        self.add_subsystem(
            "aero_slipstream_openvsp",
            _ComputeSlipstreamOpenvsp(
                propulsion_id=self.options["propulsion_id"],
                result_folder_path=self.options["result_folder_path"],
                openvsp_exe_path=self.options["openvsp_exe_path"],
                airfoil_folder_path=self.options["airfoil_folder_path"],
                wing_airfoil_file=self.options["wing_airfoil_file"],
                low_speed_aero=self.options["low_speed_aero"],
            ),
            promotes=["data:*"],
        )


class _ComputeSlipstreamOpenvsp(OPENVSPSimpleGeometryDP):
    def initialize(self):
        super().initialize()
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        super().setup()
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan, units="deg**-1")
        else:
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="deg**-1")

        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")

        self.add_input("thrust", val=np.nan, units="N")
        self.add_input("shaft_power", val=np.nan, units="W")

        if self.options["low_speed_aero"]:
            self.add_output(
                "data:aerodynamics:slipstream:wing:low_speed:prop_on:Y_vector",
                shape=SPAN_MESH_POINT,
                units="m",
            )
            self.add_output(
                "data:aerodynamics:slipstream:wing:low_speed:prop_on:CL_vector",
                shape=SPAN_MESH_POINT,
            )
            self.add_output("data:aerodynamics:slipstream:wing:low_speed:prop_on:CT_ref")
            self.add_output("data:aerodynamics:slipstream:wing:low_speed:prop_on:CL")
            self.add_output(
                "data:aerodynamics:slipstream:wing:low_speed:prop_on:velocity", units="m/s"
            )
            self.add_output(
                "data:aerodynamics:slipstream:wing:low_speed:prop_off:Y_vector",
                shape=SPAN_MESH_POINT,
                units="m",
            )
            self.add_output(
                "data:aerodynamics:slipstream:wing:low_speed:prop_off:CL_vector",
                shape=SPAN_MESH_POINT,
            )
            self.add_output("data:aerodynamics:slipstream:wing:low_speed:prop_off:CL")
            self.add_output(
                "data:aerodynamics:slipstream:wing:low_speed:only_prop:CL_vector",
                shape=SPAN_MESH_POINT,
            )
        else:
            self.add_output(
                "data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector",
                shape=SPAN_MESH_POINT,
                units="m",
            )
            self.add_output(
                "data:aerodynamics:slipstream:wing:cruise:prop_on:CL_vector", shape=SPAN_MESH_POINT
            )
            self.add_output("data:aerodynamics:slipstream:wing:cruise:prop_on:CT_ref")
            self.add_output("data:aerodynamics:slipstream:wing:cruise:prop_on:CL")
            self.add_output(
                "data:aerodynamics:slipstream:wing:cruise:prop_on:velocity", units="m/s"
            )
            self.add_output(
                "data:aerodynamics:slipstream:wing:cruise:prop_off:Y_vector",
                shape=SPAN_MESH_POINT,
                units="m",
            )
            self.add_output(
                "data:aerodynamics:slipstream:wing:cruise:prop_off:CL_vector", shape=SPAN_MESH_POINT
            )
            self.add_output("data:aerodynamics:slipstream:wing:cruise:prop_off:CL")
            self.add_output(
                "data:aerodynamics:slipstream:wing:cruise:only_prop:CL_vector",
                shape=SPAN_MESH_POINT,
            )

    def check_config(self, logger):
        # let void to avoid logger error on "The command cannot be empty"
        pass

    def compute(self, inputs, outputs):

        if self.options["low_speed_aero"]:
            altitude = 0.0
            mach = inputs["data:aerodynamics:low_speed:mach"]
            cl0 = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
            cl_alpha = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
        else:
            altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
            mach = inputs["data:aerodynamics:cruise:mach"]
            cl0 = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
            cl_alpha = inputs["data:aerodynamics:wing:cruise:CL_alpha"]

        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]

        atm = Atmosphere(altitude, altitude_in_feet=False)
        velocity = mach * atm.speed_of_sound

        # We need to compute the AOA for which the most constraining Delta_Cl due to slipstream
        # will appear, this is taken as the angle for which the clean wing is at its max angle of
        # attack

        alpha_max = (cl_max_clean - cl0) / cl_alpha

        wing_rotor = self.compute_wing_rotor(
            inputs, outputs, altitude, mach, alpha_max, inputs["thrust"], inputs["shaft_power"]
        )
        wing = self.compute_wing(inputs, outputs, altitude, mach, alpha_max)

        cl_vector_prop_on = wing_rotor["cl_vector"]
        y_vector_prop_on = wing_rotor["y_vector"]

        cl_vector_prop_off = wing["cl_vector"]
        y_vector_prop_off = wing["y_vector"]

        additional_zeros = list(np.zeros(SPAN_MESH_POINT - len(cl_vector_prop_on)))
        cl_vector_prop_on.extend(additional_zeros)
        y_vector_prop_on.extend(additional_zeros)
        cl_vector_prop_off.extend(additional_zeros)
        y_vector_prop_off.extend(additional_zeros)

        cl_diff = []
        for i in range(len(cl_vector_prop_on)):
            cl_diff.append(round(cl_vector_prop_on[i] - cl_vector_prop_off[i], 4))

        if self.options["low_speed_aero"]:
            outputs[
                "data:aerodynamics:slipstream:wing:low_speed:prop_on:Y_vector"
            ] = y_vector_prop_on
            outputs[
                "data:aerodynamics:slipstream:wing:low_speed:prop_on:CL_vector"
            ] = cl_vector_prop_on
            outputs["data:aerodynamics:slipstream:wing:low_speed:prop_on:CT_ref"] = wing_rotor["ct"]
            outputs["data:aerodynamics:slipstream:wing:low_speed:prop_on:CL"] = wing_rotor["cl"]
            outputs["data:aerodynamics:slipstream:wing:low_speed:prop_on:velocity"] = velocity
            outputs[
                "data:aerodynamics:slipstream:wing:low_speed:prop_off:Y_vector"
            ] = y_vector_prop_off
            outputs[
                "data:aerodynamics:slipstream:wing:low_speed:prop_off:CL_vector"
            ] = cl_vector_prop_off
            outputs["data:aerodynamics:slipstream:wing:low_speed:prop_off:CL"] = wing["cl"]
            outputs["data:aerodynamics:slipstream:wing:low_speed:only_prop:CL_vector"] = cl_diff
        else:
            outputs["data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector"] = y_vector_prop_on
            outputs[
                "data:aerodynamics:slipstream:wing:cruise:prop_on:CL_vector"
            ] = cl_vector_prop_on
            outputs["data:aerodynamics:slipstream:wing:cruise:prop_on:CT_ref"] = wing_rotor["ct"]
            outputs["data:aerodynamics:slipstream:wing:cruise:prop_on:CL"] = wing_rotor["cl"]
            outputs["data:aerodynamics:slipstream:wing:cruise:prop_on:velocity"] = velocity
            outputs[
                "data:aerodynamics:slipstream:wing:cruise:prop_off:Y_vector"
            ] = y_vector_prop_off
            outputs[
                "data:aerodynamics:slipstream:wing:cruise:prop_off:CL_vector"
            ] = cl_vector_prop_off
            outputs["data:aerodynamics:slipstream:wing:cruise:prop_off:CL"] = wing["cl"]
            outputs["data:aerodynamics:slipstream:wing:cruise:only_prop:CL_vector"] = cl_diff


class FlightConditionsForDPComputation(om.ExplicitComponent):
    """
    Makes the flight conditions for the thrust and power computation locally available for the
    slipstream computation.
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
        else:
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")

        self.add_output("mach")
        self.add_output("altitude", units="m")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        if self.options["low_speed_aero"]:
            mach = inputs["data:aerodynamics:low_speed:mach"]
            altitude = 0.0
        else:
            mach = inputs["data:aerodynamics:cruise:mach"]
            altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]

        outputs["mach"] = mach
        outputs["altitude"] = altitude


@oad.RegisterSubmodel(
    SUBMODEL_THRUST_POWER_SLIPSTREAM,
    "fastga.submodel.aerodynamics.wing.slipstream.thrust_power_computation.via_id",
)
class PropulsionForDPComputation(om.ExplicitComponent):
    """Computes thrust and shaft power for slisptream computation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):

        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)
        self.add_input("mach", val=np.nan)
        self.add_input("altitude", val=np.nan, units="m")

        self.add_output("thrust", val=0, units="N")
        self.add_output("shaft_power", val=1, units="W")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propulsion_model = self._engine_wrapper.get_model(inputs)
        flight_point = oad.FlightPoint(
            mach=inputs["mach"],
            altitude=inputs["altitude"],
            engine_setting=EngineSetting.CLIMB,
            thrust_rate=1.0,
        )
        propulsion_model.compute_flight_points(flight_point)
        thrust = float(flight_point.thrust)

        thrust_one_prop = thrust / inputs["data:geometry:propulsion:engine:count"]
        atm = Atmosphere(inputs["altitude"], altitude_in_feet=False)
        atm.mach = inputs["mach"]

        propeller_efficiency = float(
            propulsion_model.engine.propeller_efficiency(thrust_one_prop, atm)
        )

        outputs["thrust"] = thrust
        outputs["shaft_power"] = thrust * atm.true_airspeed / propeller_efficiency

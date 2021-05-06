"""
    Estimation of slipstream effects using OPENVSP
"""
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
from openmdao.core.group import Group
from fastoad.model_base import Atmosphere

from .openvsp import OPENVSPSimpleGeometryDPX57, DEFAULT_WING_AIRFOIL
from ...constants import SPAN_MESH_POINT
from ...components.compute_reynolds import ComputeUnitReynolds

PROPELLER_NUMBER = 7


class ComputeSlipstreamOpenvspX57(Group):

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare("openvsp_exe_path", default="", types=str, allow_none=True)
        self.options.declare("wing_airfoil_file", default=DEFAULT_WING_AIRFOIL, types=str, allow_none=True)

    def setup(self):
        self.add_subsystem("comp_unit_reynolds_slipstream", ComputeUnitReynolds(
            low_speed_aero=False), promotes=["*"])
        self.add_subsystem("aero_slipstream_openvsp_x57",
                           _ComputeSlipstreamOpenvspX57(
                               propulsion_id=self.options["propulsion_id"],
                               result_folder_path=self.options["result_folder_path"],
                               wing_airfoil_file=self.options["wing_airfoil_file"],
                           ), promotes=["*"])


class _ComputeSlipstreamOpenvspX57(OPENVSPSimpleGeometryDPX57):

    def setup(self):
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_0", val=np.nan, units="deg")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")

        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")

        self.add_input("data:aerodynamics:cruise:mach", val=np.nan)

        self.add_output("data:aerodynamics:slipstream:wing:prop_on:Y_vector", shape=SPAN_MESH_POINT,
                        units="m")
        self.add_output("data:aerodynamics:slipstream:wing:prop_on:CL_vector", shape=SPAN_MESH_POINT)
        self.add_output("data:aerodynamics:slipstream:wing:prop_on:CT_ref", shape=PROPELLER_NUMBER)
        self.add_output("data:aerodynamics:slipstream:wing:prop_on:CL")
        self.add_output("data:aerodynamics:slipstream:wing:prop_on:velocity", units="m/s")
        self.add_output("data:aerodynamics:slipstream:wing:prop_off:Y_vector", shape=SPAN_MESH_POINT,
                        units="m")
        self.add_output("data:aerodynamics:slipstream:wing:prop_off:CL_vector", shape=SPAN_MESH_POINT)
        self.add_output("data:aerodynamics:slipstream:wing:prop_off:CL")
        self.add_output("data:aerodynamics:slipstream:wing:only_prop:CL_vector", shape=SPAN_MESH_POINT)


    def check_config(self, logger):
        # let void to avoid logger error on "The command cannot be empty"
        pass

    def compute(self, inputs, outputs):

        altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        mach = inputs["data:aerodynamics:cruise:mach"]

        atm = Atmosphere(altitude, altitude_in_feet=False)
        velocity = mach * atm.speed_of_sound

        # We need to compute the AOA for which the most constraining Delta_Cl due to slipstream will appear, this
        # is taken as the angle for which the clean wing is at its max angle of attack

        alpha_max = 15.0

        wing_rotor = self.compute_wing_rotor_x57(inputs, outputs, altitude, mach, alpha_max)
        wing = self.compute_wing(inputs, outputs, altitude, mach, alpha_max)

        cl_vector_prop_on = wing_rotor["cl_vector"]
        y_vector_prop_on = wing_rotor["y_vector"]

        cl_vector_prop_off = wing["cl_vector"]
        y_vector_prop_off = wing["y_vector"]

        additional_zeros = list(np.zeros(SPAN_MESH_POINT-len(cl_vector_prop_on)))
        cl_vector_prop_on.extend(additional_zeros)
        y_vector_prop_on.extend(additional_zeros)
        cl_vector_prop_off.extend(additional_zeros)
        y_vector_prop_off.extend(additional_zeros)

        cl_diff = []
        for i in range(len(cl_vector_prop_on)):
            cl_diff.append(round(cl_vector_prop_on[i]-cl_vector_prop_off[i], 4))

        outputs["data:aerodynamics:slipstream:wing:prop_on:Y_vector"] = y_vector_prop_on
        outputs["data:aerodynamics:slipstream:wing:prop_on:CL_vector"] = cl_vector_prop_on
        outputs["data:aerodynamics:slipstream:wing:prop_on:CT_ref"] = wing_rotor["ct"]
        outputs["data:aerodynamics:slipstream:wing:prop_on:CL"] = wing_rotor["cl"]
        outputs["data:aerodynamics:slipstream:wing:prop_on:velocity"] = velocity

        outputs["data:aerodynamics:slipstream:wing:prop_off:Y_vector"] = y_vector_prop_off
        outputs["data:aerodynamics:slipstream:wing:prop_off:CL_vector"] = cl_vector_prop_off
        outputs["data:aerodynamics:slipstream:wing:prop_off:CL"] = wing["cl"]

        outputs["data:aerodynamics:slipstream:wing:only_prop:CL_vector"] = cl_diff

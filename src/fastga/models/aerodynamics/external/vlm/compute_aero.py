"""
    Estimation of cl/cm/oswald aero coefficients using VLM.
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

import logging

import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.group import Group

from .vlm import VLMSimpleGeometry
from ..xfoil.xfoil_polar import XfoilPolar
from ...components.compute_reynolds import ComputeUnitReynolds
from ...constants import SPAN_MESH_POINT, MACH_NB_PTS

_LOGGER = logging.getLogger(__name__)

DEFAULT_WING_AIRFOIL = "naca23012.af"
DEFAULT_HTP_AIRFOIL = "naca0012.af"
INPUT_AOA = 10.0  # only one value given since calculation is done by default around 0.0!


class ComputeAEROvlm(Group):
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare("compute_mach_interpolation", default=False, types=bool)
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare(
            "wing_airfoil_file", default=DEFAULT_WING_AIRFOIL, types=str, allow_none=True
        )
        self.options.declare(
            "htp_airfoil_file", default=DEFAULT_HTP_AIRFOIL, types=str, allow_none=True
        )

    def setup(self):
        self.add_subsystem(
            "comp_unit_reynolds",
            ComputeUnitReynolds(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["*"],
        )
        self.add_subsystem(
            "comp_local_reynolds",
            ComputeLocalReynolds(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["*"],
        )
        if self.options["low_speed_aero"]:
            self.add_subsystem(
                "wing_polar_ls",
                XfoilPolar(
                    airfoil_folder_path=self.options["airfoil_folder_path"],
                    airfoil_file=self.options["wing_airfoil_file"],
                    alpha_end=20.0,
                    activate_negative_angle=True,
                ),
                promotes=[],
            )
            self.add_subsystem(
                "htp_polar_ls",
                XfoilPolar(
                    airfoil_folder_path=self.options["airfoil_folder_path"],
                    airfoil_file=self.options["htp_airfoil_file"],
                    alpha_end=20.0,
                    activate_negative_angle=True,
                ),
                promotes=[],
            )
        else:
            self.add_subsystem(
                "wing_polar_hs",
                XfoilPolar(
                    airfoil_folder_path=self.options["airfoil_folder_path"],
                    airfoil_file=self.options["wing_airfoil_file"],
                    alpha_end=20.0,
                    activate_negative_angle=True,
                ),
                promotes=[],
            )
            self.add_subsystem(
                "htp_polar_hs",
                XfoilPolar(
                    airfoil_folder_path=self.options["airfoil_folder_path"],
                    airfoil_file=self.options["htp_airfoil_file"],
                    alpha_end=20.0,
                    activate_negative_angle=True,
                ),
                promotes=[],
            )
        self.add_subsystem(
            "aero_vlm",
            _ComputeAEROvlm(
                low_speed_aero=self.options["low_speed_aero"],
                result_folder_path=self.options["result_folder_path"],
                compute_mach_interpolation=self.options["compute_mach_interpolation"],
                airfoil_folder_path=self.options["airfoil_folder_path"],
                wing_airfoil_file=self.options["wing_airfoil_file"],
                htp_airfoil_file=self.options["htp_airfoil_file"],
            ),
            promotes=["*"],
        )

        if self.options["low_speed_aero"]:
            self.connect("data:aerodynamics:low_speed:mach", "wing_polar_ls.xfoil:mach")
            self.connect(
                "data:aerodynamics:wing:low_speed:reynolds", "wing_polar_ls.xfoil:reynolds"
            )
            self.connect("wing_polar_ls.xfoil:CL", "data:aerodynamics:wing:low_speed:CL")
            self.connect("wing_polar_ls.xfoil:CDp", "data:aerodynamics:wing:low_speed:CDp")
            self.connect("data:aerodynamics:low_speed:mach", "htp_polar_ls.xfoil:mach")
            self.connect(
                "data:aerodynamics:horizontal_tail:low_speed:reynolds",
                "htp_polar_ls.xfoil:reynolds",
            )
            self.connect("htp_polar_ls.xfoil:CL", "data:aerodynamics:horizontal_tail:low_speed:CL")
            self.connect(
                "htp_polar_ls.xfoil:CDp", "data:aerodynamics:horizontal_tail:low_speed:CDp"
            )
        else:
            self.connect("data:aerodynamics:cruise:mach", "wing_polar_hs.xfoil:mach")
            self.connect("data:aerodynamics:wing:cruise:reynolds", "wing_polar_hs.xfoil:reynolds")
            self.connect("wing_polar_hs.xfoil:CL", "data:aerodynamics:wing:cruise:CL")
            self.connect("wing_polar_hs.xfoil:CDp", "data:aerodynamics:wing:cruise:CDp")
            self.connect("data:aerodynamics:cruise:mach", "htp_polar_hs.xfoil:mach")
            self.connect(
                "data:aerodynamics:horizontal_tail:cruise:reynolds", "htp_polar_hs.xfoil:reynolds"
            )
            self.connect("htp_polar_hs.xfoil:CL", "data:aerodynamics:horizontal_tail:cruise:CL")
            self.connect("htp_polar_hs.xfoil:CDp", "data:aerodynamics:horizontal_tail:cruise:CDp")


class ComputeLocalReynolds(ExplicitComponent):
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:unit_reynolds", val=np.nan, units="m**-1")
        else:
            self.add_input("data:aerodynamics:cruise:unit_reynolds", val=np.nan, units="m**-1")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")

        if self.options["low_speed_aero"]:
            self.add_output("data:aerodynamics:wing:low_speed:reynolds")
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:reynolds")
        else:
            self.add_output("data:aerodynamics:wing:cruise:reynolds")
            self.add_output("data:aerodynamics:horizontal_tail:cruise:reynolds")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:wing:low_speed:reynolds"] = (
                inputs["data:aerodynamics:low_speed:unit_reynolds"]
                * inputs["data:geometry:wing:MAC:length"]
            )
            outputs["data:aerodynamics:horizontal_tail:low_speed:reynolds"] = (
                inputs["data:aerodynamics:low_speed:unit_reynolds"]
                * inputs["data:geometry:horizontal_tail:MAC:length"]
            )
        else:
            outputs["data:aerodynamics:wing:cruise:reynolds"] = (
                inputs["data:aerodynamics:cruise:unit_reynolds"]
                * inputs["data:geometry:wing:MAC:length"]
            )
            outputs["data:aerodynamics:horizontal_tail:cruise:reynolds"] = (
                inputs["data:aerodynamics:cruise:unit_reynolds"]
                * inputs["data:geometry:horizontal_tail:MAC:length"]
            )


class _ComputeAEROvlm(VLMSimpleGeometry):
    def initialize(self):
        super().initialize()
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare("compute_mach_interpolation", default=False, types=bool)

    def setup(self):

        super().setup()
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
        else:
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")

        if self.options["low_speed_aero"]:
            self.add_output("data:aerodynamics:wing:low_speed:CL0_clean")
            self.add_output("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:wing:low_speed:CM0_clean")
            self.add_output(
                "data:aerodynamics:wing:low_speed:Y_vector", shape=SPAN_MESH_POINT, units="m"
            )
            self.add_output("data:aerodynamics:wing:low_speed:CL_vector", shape=SPAN_MESH_POINT)
            self.add_output(
                "data:aerodynamics:wing:low_speed:chord_vector", shape=SPAN_MESH_POINT, units="m"
            )
            self.add_output("data:aerodynamics:wing:low_speed:induced_drag_coefficient")
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:CL0")
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_ref")
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
            self.add_output(
                "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated", units="rad**-1"
            )
            self.add_output(
                "data:aerodynamics:horizontal_tail:low_speed:Y_vector",
                shape=SPAN_MESH_POINT,
                units="m",
            )
            self.add_output(
                "data:aerodynamics:horizontal_tail:low_speed:CL_vector", shape=SPAN_MESH_POINT
            )
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient")
        else:
            self.add_output("data:aerodynamics:wing:cruise:CL0_clean")
            self.add_output("data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:wing:cruise:CM0_clean")
            self.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient")
            self.add_output("data:aerodynamics:horizontal_tail:cruise:CL0")
            self.add_output("data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1")
            self.add_output(
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated", units="rad**-1"
            )
            self.add_output("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient")
            if self.options["compute_mach_interpolation"]:
                self.add_output(
                    "data:aerodynamics:aircraft:mach_interpolation:mach_vector",
                    shape=MACH_NB_PTS + 1,
                )
                self.add_output(
                    "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector",
                    shape=MACH_NB_PTS + 1,
                    units="rad**-1",
                )

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        _LOGGER.debug("Entering aerodynamic computation")

        # Check AOA input is float
        if not isinstance(INPUT_AOA, float):
            raise TypeError("INPUT_AOA should be a float!")

        if self.options["low_speed_aero"]:
            altitude = 0.0
            mach = inputs["data:aerodynamics:low_speed:mach"]
        else:
            altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
            mach = inputs["data:aerodynamics:cruise:mach"]

        (
            cl_0_wing,
            cl_alpha_wing,
            cm_0_wing,
            y_vector_wing,
            cl_vector_wing,
            chord_vector_wing,
            coef_k_wing,
            cl_0_htp,
            cl_X_htp,
            cl_alpha_htp,
            cl_alpha_htp_isolated,
            y_vector_htp,
            cl_vector_htp,
            coef_k_htp,
        ) = self.compute_aero_coeff(inputs, altitude, mach, INPUT_AOA)

        if self.options["low_speed_aero"]:
            pass
        else:
            if self.options["compute_mach_interpolation"]:
                mach_interp, cl_alpha_interp = self.compute_cl_alpha_mach(
                    inputs, INPUT_AOA, altitude, mach
                )

        # Defining outputs
        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:wing:low_speed:CL0_clean"] = cl_0_wing
            outputs["data:aerodynamics:wing:low_speed:CL_alpha"] = cl_alpha_wing
            outputs["data:aerodynamics:wing:low_speed:CM0_clean"] = cm_0_wing
            outputs["data:aerodynamics:wing:low_speed:Y_vector"] = y_vector_wing
            outputs["data:aerodynamics:wing:low_speed:CL_vector"] = cl_vector_wing
            outputs["data:aerodynamics:wing:low_speed:chord_vector"] = chord_vector_wing
            outputs["data:aerodynamics:wing:low_speed:induced_drag_coefficient"] = coef_k_wing
            outputs["data:aerodynamics:horizontal_tail:low_speed:CL0"] = cl_0_htp
            outputs["data:aerodynamics:horizontal_tail:low_speed:CL_ref"] = cl_X_htp
            outputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"] = cl_alpha_htp
            outputs[
                "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated"
            ] = cl_alpha_htp_isolated
            outputs["data:aerodynamics:horizontal_tail:low_speed:Y_vector"] = y_vector_htp
            outputs["data:aerodynamics:horizontal_tail:low_speed:CL_vector"] = cl_vector_htp
            outputs[
                "data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"
            ] = coef_k_htp
        else:
            outputs["data:aerodynamics:wing:cruise:CL0_clean"] = cl_0_wing
            outputs["data:aerodynamics:wing:cruise:CL_alpha"] = cl_alpha_wing
            outputs["data:aerodynamics:wing:cruise:CM0_clean"] = cm_0_wing
            outputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"] = coef_k_wing
            outputs["data:aerodynamics:horizontal_tail:cruise:CL0"] = cl_0_htp
            outputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"] = cl_alpha_htp
            outputs[
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated"
            ] = cl_alpha_htp_isolated
            outputs[
                "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"
            ] = coef_k_htp
            if self.options["compute_mach_interpolation"]:
                outputs["data:aerodynamics:aircraft:mach_interpolation:mach_vector"] = mach_interp
                outputs[
                    "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"
                ] = cl_alpha_interp

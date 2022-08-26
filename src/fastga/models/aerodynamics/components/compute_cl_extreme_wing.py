"""Estimation of the 3D maximum lift coefficients for clean wing."""
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
import fastoad.api as oad

from fastga.models.aerodynamics.constants import SPAN_MESH_POINT, SUBMODEL_CL_EXTREME_CLEAN_WING
from fastga.models.aerodynamics.external.xfoil.xfoil_polar import XfoilPolar


@oad.RegisterSubmodel(
    SUBMODEL_CL_EXTREME_CLEAN_WING,
    "fastga.submodel.aerodynamics.wing.extreme_lift_coefficient.clean.legacy",
)
class ComputeExtremeCLWing(om.Group):
    """
    Computes maximum CL of the wing in clean configuration.

    3D CL is deduced from 2D CL asymptote and the hypothesis that the max 3D lift corresponds to
    the lift at which one section of the wing goes out of the linear range in its lift curve
    slope.
    """

    def initialize(self):
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare(
            "wing_airfoil_file", default="naca23012.af", types=str, allow_none=True
        )

    def setup(self):
        self.add_subsystem(
            "comp_local_reynolds_wing",
            ComputeLocalReynolds(),
            promotes=[
                "data:aerodynamics:low_speed:mach",
                "data:aerodynamics:low_speed:unit_reynolds",
                "data:geometry:wing:root:chord",
                "data:geometry:wing:tip:chord",
                "data:aerodynamics:wing:root:low_speed:reynolds",
                "data:aerodynamics:wing:tip:low_speed:reynolds",
            ],
        )
        self.add_subsystem(
            "wing_root_polar",
            XfoilPolar(
                airfoil_folder_path=self.options["airfoil_folder_path"],
                alpha_end=20.0,
                airfoil_file=self.options["wing_airfoil_file"],
                activate_negative_angle=True,
            ),
            promotes=[],
        )
        self.add_subsystem(
            "wing_tip_polar",
            XfoilPolar(
                airfoil_folder_path=self.options["airfoil_folder_path"],
                alpha_end=20.0,
                airfoil_file=self.options["wing_airfoil_file"],
                activate_negative_angle=True,
            ),
            promotes=[],
        )

        self.add_subsystem("CL_3D_wing", ComputeWing3DExtremeCL(), promotes=["*"])

        self.connect("comp_local_reynolds_wing.xfoil:mach", "wing_root_polar.xfoil:mach")
        self.connect(
            "data:aerodynamics:wing:root:low_speed:reynolds", "wing_root_polar.xfoil:reynolds"
        )
        self.connect(
            "wing_root_polar.xfoil:CL_max_2D", "data:aerodynamics:wing:low_speed:root:CL_max_2D"
        )
        self.connect(
            "wing_root_polar.xfoil:CL_min_2D", "data:aerodynamics:wing:low_speed:root:CL_min_2D"
        )

        self.connect("comp_local_reynolds_wing.xfoil:mach", "wing_tip_polar.xfoil:mach")
        self.connect(
            "data:aerodynamics:wing:tip:low_speed:reynolds", "wing_tip_polar.xfoil:reynolds"
        )
        self.connect(
            "wing_tip_polar.xfoil:CL_max_2D", "data:aerodynamics:wing:low_speed:tip:CL_max_2D"
        )
        self.connect(
            "wing_tip_polar.xfoil:CL_min_2D", "data:aerodynamics:wing:low_speed:tip:CL_min_2D"
        )


class ComputeLocalReynolds(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
        self.add_input("data:aerodynamics:low_speed:unit_reynolds", val=np.nan, units="m**-1")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")

        self.add_output("data:aerodynamics:wing:root:low_speed:reynolds")
        self.add_output("data:aerodynamics:wing:tip:low_speed:reynolds")
        self.add_output("xfoil:mach")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:aerodynamics:wing:root:low_speed:reynolds"] = (
            inputs["data:aerodynamics:low_speed:unit_reynolds"]
            * inputs["data:geometry:wing:root:chord"]
        )
        outputs["data:aerodynamics:wing:tip:low_speed:reynolds"] = (
            inputs["data:aerodynamics:low_speed:unit_reynolds"]
            * inputs["data:geometry:wing:tip:chord"]
        )
        outputs["xfoil:mach"] = inputs["data:aerodynamics:low_speed:mach"]


class ComputeWing3DExtremeCL(om.ExplicitComponent):
    """Computes wing 3D min/max CL from 2D CL (XFOIL-computed) and lift repartition."""

    def setup(self):

        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:aerodynamics:wing:low_speed:root:CL_max_2D", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:tip:CL_max_2D", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:root:CL_min_2D", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:tip:CL_min_2D", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
        self.add_input(
            "data:aerodynamics:wing:low_speed:Y_vector",
            val=np.nan,
            shape_by_conn=True,
            units="m",
        )
        self.add_input(
            "data:aerodynamics:wing:low_speed:CL_vector",
            val=np.nan,
            shape_by_conn=True,
            copy_shape="data:aerodynamics:wing:low_speed:Y_vector",
        )

        self.add_output("data:aerodynamics:wing:low_speed:CL_max_clean")
        self.add_output("data:aerodynamics:wing:low_speed:CL_min_clean")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        y_root = float(inputs["data:geometry:wing:root:y"])
        y_tip = float(inputs["data:geometry:wing:tip:y"])
        cl_max_2d_root = float(inputs["data:aerodynamics:wing:low_speed:root:CL_max_2D"])
        cl_max_2d_tip = float(inputs["data:aerodynamics:wing:low_speed:tip:CL_max_2D"])
        cl_min_2d_root = float(inputs["data:aerodynamics:wing:low_speed:root:CL_min_2D"])
        cl_min_2d_tip = float(inputs["data:aerodynamics:wing:low_speed:tip:CL_min_2D"])
        cl0 = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
        y_interp = inputs["data:aerodynamics:wing:low_speed:Y_vector"]
        cl_interp = inputs["data:aerodynamics:wing:low_speed:CL_vector"]

        y_interp, cl_interp = self._reshape_curve(y_interp, cl_interp)
        y_vector = np.linspace(
            max(y_root, min(y_interp)), min(y_tip, max(y_interp)), SPAN_MESH_POINT
        )
        cl_xfoil_max = np.interp(
            y_vector, np.array([y_root, y_tip]), np.array([cl_max_2d_root, cl_max_2d_tip])
        )
        cl_xfoil_min = np.interp(
            y_vector, np.array([y_root, y_tip]), np.array([cl_min_2d_root, cl_min_2d_tip])
        )
        cl_curve = np.maximum(
            np.interp(y_vector, y_interp, cl_interp), 1e-12 * np.ones(np.size(y_vector))
        )  # avoid divide by 0
        cl_max_clean = cl0 * np.min(cl_xfoil_max / cl_curve)
        cl_min_clean = cl0 * np.max(cl_xfoil_min / cl_curve)

        outputs["data:aerodynamics:wing:low_speed:CL_max_clean"] = cl_max_clean
        outputs["data:aerodynamics:wing:low_speed:CL_min_clean"] = cl_min_clean

    @staticmethod
    def _reshape_curve(y: np.ndarray, cl: np.ndarray):
        """Reshape data from openvsp/vlm lift curve"""

        for idx in range(len(y)):
            if np.sum(y[idx : len(y)] == 0) == (len(y) - idx):
                y = y[0:idx]
                cl = cl[0:idx]
                break

        return y, cl

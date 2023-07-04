"""Estimation of the 3D maximum lift coefficients for clean tail."""
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

from fastga.models.aerodynamics.constants import SPAN_MESH_POINT, SUBMODEL_CL_EXTREME_CLEAN_HT
from fastga.models.aerodynamics.external.xfoil.xfoil_polar import XfoilPolar


@oad.RegisterSubmodel(
    SUBMODEL_CL_EXTREME_CLEAN_HT,
    "fastga.submodel.aerodynamics.horizontal_tail.extreme_lift_coefficient.clean.legacy",
)
class ComputeExtremeCLHtp(om.Group):
    """
    Computes maximum CL of the horizontal tail plane in clean configuration.

    3D CL is deduced from 2D CL asymptote and the hypothesis that the max 3D lift corresponds to
    the lift at which one section of the htp goes out of the linear range in its lift curve
    slope.
    """

    def initialize(self):
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare("htp_airfoil_file", default="naca0012.af", types=str, allow_none=True)

    def setup(self):
        self.add_subsystem(
            "comp_local_reynolds_htp",
            ComputeLocalReynolds(),
            promotes=[
                "data:aerodynamics:low_speed:mach",
                "data:aerodynamics:low_speed:unit_reynolds",
                "data:geometry:horizontal_tail:root:chord",
                "data:geometry:horizontal_tail:tip:chord",
                "data:aerodynamics:horizontal_tail:efficiency",
                "data:aerodynamics:horizontal_tail:root:low_speed:reynolds",
                "data:aerodynamics:horizontal_tail:tip:low_speed:reynolds",
            ],
        )
        self.add_subsystem(
            "htp_root_polar",
            XfoilPolar(
                airfoil_folder_path=self.options["airfoil_folder_path"],
                alpha_end=20.0,
                airfoil_file=self.options["htp_airfoil_file"],
                activate_negative_angle=True,
            ),
            promotes=[],
        )
        self.add_subsystem(
            "htp_tip_polar",
            XfoilPolar(
                airfoil_folder_path=self.options["airfoil_folder_path"],
                alpha_end=20.0,
                airfoil_file=self.options["htp_airfoil_file"],
                activate_negative_angle=True,
            ),
            promotes=[],
        )
        self.add_subsystem("CL_3D_htp", ComputeHtp3DExtremeCL(), promotes=["*"])

        self.connect("comp_local_reynolds_htp.xfoil:mach", "htp_root_polar.xfoil:mach")
        self.connect(
            "data:aerodynamics:horizontal_tail:root:low_speed:reynolds",
            "htp_root_polar.xfoil:reynolds",
        )
        self.connect(
            "htp_root_polar.xfoil:CL_max_2D",
            "data:aerodynamics:horizontal_tail:low_speed:root:CL_max_2D",
        )
        self.connect(
            "htp_root_polar.xfoil:CL_min_2D",
            "data:aerodynamics:horizontal_tail:low_speed:root:CL_min_2D",
        )

        self.connect("comp_local_reynolds_htp.xfoil:mach", "htp_tip_polar.xfoil:mach")
        self.connect(
            "data:aerodynamics:horizontal_tail:tip:low_speed:reynolds",
            "htp_tip_polar.xfoil:reynolds",
        )
        self.connect(
            "htp_tip_polar.xfoil:CL_max_2D",
            "data:aerodynamics:horizontal_tail:low_speed:tip:CL_max_2D",
        )
        self.connect(
            "htp_tip_polar.xfoil:CL_min_2D",
            "data:aerodynamics:horizontal_tail:low_speed:tip:CL_min_2D",
        )


class ComputeLocalReynolds(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
        self.add_input("data:aerodynamics:low_speed:unit_reynolds", val=np.nan, units="m**-1")
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:aerodynamics:horizontal_tail:efficiency", val=0.9)

        self.add_output("data:aerodynamics:horizontal_tail:root:low_speed:reynolds")
        self.add_output("data:aerodynamics:horizontal_tail:tip:low_speed:reynolds")
        self.add_output("xfoil:mach")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:aerodynamics:horizontal_tail:root:low_speed:reynolds"] = (
            inputs["data:aerodynamics:low_speed:unit_reynolds"]
            * inputs["data:geometry:horizontal_tail:root:chord"]
            * np.sqrt(inputs["data:aerodynamics:horizontal_tail:efficiency"])
        )
        outputs["data:aerodynamics:horizontal_tail:tip:low_speed:reynolds"] = (
            inputs["data:aerodynamics:low_speed:unit_reynolds"]
            * inputs["data:geometry:horizontal_tail:tip:chord"]
            * np.sqrt(inputs["data:aerodynamics:horizontal_tail:efficiency"])
        )
        outputs["xfoil:mach"] = inputs["data:aerodynamics:low_speed:mach"]


class ComputeHtp3DExtremeCL(om.ExplicitComponent):
    """Computes HTP 3D min/max CL from 2D CL (XFOIL-computed) and lift repartition."""

    def setup(self):

        nans_array = np.full(SPAN_MESH_POINT, np.nan)
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:tip:CL_max_2D", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:root:CL_max_2D", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:tip:CL_min_2D", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:root:CL_min_2D", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL_ref", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:Y_vector", val=nans_array, units="m"
        )
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL_vector", val=nans_array)
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1"
        )

        self.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_max_clean")
        self.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_min_clean")
        self.add_output(
            "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_max", units="deg"
        )
        self.add_output(
            "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_min", units="deg"
        )

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        y_root = 0.0
        y_tip = float(inputs["data:geometry:horizontal_tail:span"]) / 2.0
        wing_area = inputs["data:geometry:wing:area"]
        htp_area = inputs["data:geometry:horizontal_tail:area"]
        area_ratio = float(htp_area / wing_area)

        cl_max_2d_root = (
            float(inputs["data:aerodynamics:horizontal_tail:low_speed:root:CL_max_2D"]) * area_ratio
        )
        cl_max_2d_tip = (
            float(inputs["data:aerodynamics:horizontal_tail:low_speed:tip:CL_max_2D"]) * area_ratio
        )
        cl_min_2d_root = (
            float(inputs["data:aerodynamics:horizontal_tail:low_speed:root:CL_min_2D"]) * area_ratio
        )
        cl_min_2d_tip = (
            float(inputs["data:aerodynamics:horizontal_tail:low_speed:tip:CL_min_2D"]) * area_ratio
        )
        cl_alpha_htp = float(inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"])
        cl_ref = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_ref"]
        y_interp = inputs["data:aerodynamics:horizontal_tail:low_speed:Y_vector"]
        cl_interp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_vector"]

        # According to Gudmundsson section 23.3, a safety margin of 0.2 should be taken for the
        # computation of the HTP stall but we already do something similar by taking as the
        # highest value, the first value having an error of 10% from linear behavior
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
        cl_curve = np.array(
            max(
                np.interp(y_vector, y_interp, cl_interp).tolist(),
                (1e-12 * np.ones(np.size(y_vector))).tolist(),
            )
        )  # avoid divide by 0
        cl_max_clean = cl_ref * np.min(cl_xfoil_max / cl_curve)
        cl_min_clean = cl_ref * np.max(cl_xfoil_min / cl_curve)

        outputs["data:aerodynamics:horizontal_tail:low_speed:CL_max_clean"] = cl_max_clean
        outputs["data:aerodynamics:horizontal_tail:low_speed:CL_min_clean"] = cl_min_clean

        clean_alpha_max_htp = cl_max_clean / cl_alpha_htp * 180.0 / np.pi
        clean_alpha_min_htp = cl_min_clean / cl_alpha_htp * 180.0 / np.pi

        outputs[
            "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_max"
        ] = clean_alpha_max_htp
        outputs[
            "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_min"
        ] = clean_alpha_min_htp

    @staticmethod
    def _reshape_curve(y: np.ndarray, cl: np.ndarray):
        """Reshape data from openvsp/vlm lift curve"""

        for idx in range(len(y)):
            if np.sum(y[idx : len(y)] == 0) == (len(y) - idx):
                y = y[0:idx]
                cl = cl[0:idx]
                break

        return y, cl

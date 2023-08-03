"""Vortex Lattice Method implementation."""
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

import copy
import logging
import os
import os.path as pth
import warnings
from typing import Optional

import numpy as np
import openmdao.api as om
import pandas as pd
from stdatm import Atmosphere
import numba
from numba import njit, prange
from fastga.models.geometry.profiles.get_profile import get_profile
from ...constants import SPAN_MESH_POINT, POLAR_POINT_COUNT, MACH_NB_PTS

DEFAULT_NX = 19
DEFAULT_NY1 = 3
DEFAULT_NY2 = 14

_LOGGER = logging.getLogger(__name__)


class VLMSimpleGeometry(om.ExplicitComponent):

    """Computation of the aerodynamics properties using the in-house VLM code."""

    def __init__(self, **kwargs):
        """Initializing parameters used in VLM computation."""
        super().__init__(**kwargs)
        self.wing = None
        self.htp = None
        self.n_x = None  # number of chordwise panel
        self.ny1 = None
        self.ny2 = None
        self.ny3 = None
        self.n_y = None  # number of semi-spanwise panel

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare(
            "wing_airfoil_file", default="naca23012.af", types=str, allow_none=True
        )
        self.options.declare("htp_airfoil_file", default="naca0012.af", types=str, allow_none=True)
        

    def setup(self):

        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:kink:span_ratio", val=np.nan)
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:flap:span_ratio", val=np.nan)
        self.add_input("data:geometry:flap:chord_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:wing:dihedral", val=np.nan, units="rad")
        self.add_input(
            "data:geometry:wing:twist",
            val=np.nan,
            units="rad",
            desc="Negative twist means tip AOA is smaller than root",
        )

        nans_array = np.full(POLAR_POINT_COUNT, np.nan)
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:wing:low_speed:CL", val=nans_array)
            self.add_input("data:aerodynamics:wing:low_speed:CDp", val=nans_array)
            self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL", val=nans_array)
            self.add_input("data:aerodynamics:horizontal_tail:low_speed:CDp", val=nans_array)
        else:
            self.add_input("data:aerodynamics:wing:cruise:CL", val=nans_array)
            self.add_input("data:aerodynamics:wing:cruise:CDp", val=nans_array)
            self.add_input("data:aerodynamics:horizontal_tail:cruise:CL", val=nans_array)
            self.add_input("data:aerodynamics:horizontal_tail:cruise:CDp", val=nans_array)

    def compute_cl_alpha_aircraft(self, inputs, altitude, mach, aoa_angle):
        """
        Function that perform a complete calculation of aerodynamic parameters under VLM and
        returns only the cl_alpha_aircraft parameter.

        :param inputs: input necessary to compute the aircraft aerodynamics.
        :param altitude: altitude at which the aerodynamic properties are computed, in m.
        :param mach: mach number used for the computation
        :param aoa_angle: angle of attack used in the aoa derivative computation, in deg.
        """
        (
            _,
            _,
            cl_alpha_wing,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            cl_alpha_htp,
            _,
            _,
            _,
            _,
        ) = self.compute_aero_coeff(inputs, altitude, mach, aoa_angle, comp_opt="ac")
        return float(cl_alpha_wing + cl_alpha_htp)

    def compute_cl_alpha_mach(self, inputs, aoa_angle, altitude, cruise_mach):
        """
        Function that performs multiple run of OpenVSP to get an interpolation of Cl_alpha as a
        function of Mach for later use in the computation of the V-n diagram.
        """
        mach_interp = np.log(np.linspace(np.exp(0.15), np.exp(1.55 * cruise_mach), MACH_NB_PTS))
        cl_alpha_interp = np.zeros(np.size(mach_interp))
        for idx, _ in enumerate(mach_interp):
            cl_alpha_interp[idx] = self.compute_cl_alpha_aircraft(
                inputs, altitude, mach_interp[idx], aoa_angle
            )

        # We add the case were M=0, for thoroughness and since we are in an incompressible flow,
        # the Cl_alpha is approximately the same as for the first Mach of the interpolation
        mach_interp = np.insert(mach_interp, 0, 0.0)
        cl_alpha_inc = cl_alpha_interp[0]
        cl_alpha_interp = np.insert(cl_alpha_interp, 0, cl_alpha_inc)

        return mach_interp, cl_alpha_interp

    def compute_aero_coeff(self, inputs, altitude, mach, aoa_angle, comp_opt="ac"):
        """
        Function that computes in VLM environment all the aerodynamic parameters @0° and
        aoa_angle and calculate the associated derivatives.

        @param inputs: inputs parameters defined within FAST-OAD-GA
        @param altitude: altitude for aerodynamic calculation in meters
        @param mach: air speed expressed in mach
        @param aoa_angle: air speed angle of attack with respect to aircraft
        @return: cl_0_wing, cl_alpha_wing, cm_0_wing, y_vector_wing, cl_vector_wing, coef_k_wing,
        cl_0_htp, cl_X_htp, cl_alpha_htp, cl_alpha_htp_isolated, y_vector_htp, cl_vector_htp,
        coef_k_htp parameters.

                ^
              y |                Points defining the panel
                |                are named clockwise. A(x_1,y_1), B(x_2,y_2), P(x_c,y_c)
        P3--B---|-----P4         Reference:
        |   |   |     |          John J. Bertin, Russell M. Cummings - Aerodynamics for Engineers
        |   |   |     |          p.394-398
        T1  |   +--P--T2---->
        |   |         |     x
        |   |         |
        P2--A--------P1
        """

        # Fix mach number of digits to consider similar results
        mach = round(float(mach) * 1e3) / 1e3

        # Get inputs necessary to define global geometry
        if self.options["low_speed_aero"]:
            cl_wing_airfoil = inputs["data:aerodynamics:wing:low_speed:CL"]
            cdp_wing_airfoil = inputs["data:aerodynamics:wing:low_speed:CDp"]
            cl_htp_airfoil = inputs["data:aerodynamics:horizontal_tail:low_speed:CL"]
            cdp_htp_airfoil = inputs["data:aerodynamics:horizontal_tail:low_speed:CDp"]
        else:
            cl_wing_airfoil = inputs["data:aerodynamics:wing:cruise:CL"]
            cdp_wing_airfoil = inputs["data:aerodynamics:wing:cruise:CDp"]
            cl_htp_airfoil = inputs["data:aerodynamics:horizontal_tail:cruise:CL"]
            cdp_htp_airfoil = inputs["data:aerodynamics:horizontal_tail:cruise:CDp"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        span_wing = inputs["data:geometry:wing:span"]
        sref_wing = float(inputs["data:geometry:wing:area"])
        sref_htp = float(inputs["data:geometry:horizontal_tail:area"])
        area_ratio = sref_htp / sref_wing
        sweep25_wing = float(inputs["data:geometry:wing:sweep_25"])
        taper_ratio_wing = float(inputs["data:geometry:wing:taper_ratio"])
        aspect_ratio_wing = float(inputs["data:geometry:wing:aspect_ratio"])
        sweep25_htp = float(inputs["data:geometry:horizontal_tail:sweep_25"])
        aspect_ratio_htp = float(inputs["data:geometry:horizontal_tail:aspect_ratio"])
        taper_ratio_htp = float(inputs["data:geometry:horizontal_tail:taper_ratio"])
        dihedral_angle = float(inputs["data:geometry:wing:dihedral"])
        twist_angle = float(inputs["data:geometry:wing:twist"])
        geometry_set = np.around(
            np.array(
                [
                    sweep25_wing,
                    taper_ratio_wing,
                    aspect_ratio_wing,
                    dihedral_angle,
                    twist_angle,
                    sweep25_htp,
                    taper_ratio_htp,
                    aspect_ratio_htp,
                    mach,
                    area_ratio,
                ]
            ),
            decimals=6,
        )

        # output initilaization
        cl_0_wing=[]
        cl_x_wing=[]
        cl_alpha_wing=[]
        cm_0_wing=[]
        y_vector_wing=[]
        cl_vector_wing=[]
        chord_vector_wing=[]
        coef_k_wing=[]
        cl_0_htp=[]
        cl_aoa_htp=[]
        cl_alpha_htp=[]
        cl_alpha_htp_isolated=[]
        y_vector_htp=[]
        cl_vector_htp=[]
        coef_k_htp=[]
        # Search if results already exist:
        result_folder_path = self.options["result_folder_path"]
        result_file_path = None
        saved_area_ratio = 1.0
        if result_folder_path != "":
            result_file_path, saved_area_ratio = self.search_results(
                result_folder_path, geometry_set
            )

        # If no result saved for that geometry under this mach condition, computation is done
        if result_file_path is None:

            # Create result folder first (if it must fail, let it fail as soon as possible)
            if result_folder_path != "":
                if not os.path.exists(result_folder_path):
                    os.makedirs(pth.join(result_folder_path), exist_ok=True)

            # Save the geometry (result_file_path is None entering the function)
            if self.options["result_folder_path"] != "":
                result_file_path = self.save_geometry(result_folder_path, geometry_set)

            # Compute wing alone @ 0°/X° angle of attack
            if comp_opt == "ac" or comp_opt == "wing":
                wing_0 = self.compute_wing(
                    inputs, altitude, mach, 0.0, flaps_angle=0.0, use_airfoil=True
                )
                wing_aoa = self.compute_wing(
                    inputs, altitude, mach, aoa_angle, flaps_angle=0.0, use_airfoil=True
                )

            # Compute complete aircraft @ 0°/X° angle of attack
            if comp_opt == "ac" or comp_opt == "htp":
                _, htp_0, _ = self.compute_aircraft(
                    inputs, altitude, mach, 0.0, flaps_angle=0.0, use_airfoil=True
                )
                _, htp_aoa, _ = self.compute_aircraft(
                    inputs, altitude, mach, aoa_angle, flaps_angle=0.0, use_airfoil=True
                )

            # Compute isolated HTP @ 0°/X° angle of attack
            if comp_opt == "ac" or comp_opt == "htp":
                htp_0_isolated = self.compute_htp(inputs, altitude, mach, 0.0, use_airfoil=True)
                htp_aoa_isolated = self.compute_htp(inputs, altitude, mach, aoa_angle, use_airfoil=True)
                
            # Post-process wing data ---------------------------------------------------------------
            if comp_opt == "ac" or comp_opt == "wing":
                (
                    beta,
                    cl_0_wing,
                    cl_x_wing,
                    cm_0_wing,
                    cl_alpha_wing,
                    y_vector_wing,
                    cl_vector_wing,
                    chord_vector_wing,
                    coef_k_wing,
                ) = self.post_processing_wing(
                    width_max,
                    span_wing,
                    mach,
                    dihedral_angle,
                    wing_0,
                    wing_aoa,
                    aoa_angle,
                    aspect_ratio_wing,
                    cl_wing_airfoil,
                    cdp_wing_airfoil,
                )

            # Post-process HTP-aircraft data -------------------------------------------------------
            if comp_opt == "ac" or comp_opt == "htp":
                (
                    cl_0_htp,
                    cl_aoa_htp,
                    cl_alpha_htp,
                    coef_k_htp,
                    y_vector_htp,
                    cl_vector_htp,
                ) = self.post_processing_htp_ac(
                    beta,
                    aspect_ratio_htp,
                    area_ratio,
                    mach,
                    aoa_angle,
                    cl_htp_airfoil,
                    cdp_htp_airfoil,
                    htp_0,
                    htp_aoa,
                )

            # Post-process HTP-isolated data -------------------------------------------------------
            if comp_opt == "ac" or comp_opt == "htp":
                cl_alpha_htp_isolated = (
                    float(htp_aoa_isolated["cl"] - htp_0_isolated["cl"])
                    / beta
                    * area_ratio
                    / (aoa_angle * np.pi / 180)
                )

            # Resize vectors -----------------------------------------------------------------------
            if comp_opt == "ac" or comp_opt == "wing":
                (y_vector_wing, cl_vector_wing, chord_vector_wing) = self.resize_wing_vector(
                    y_vector_wing, cl_vector_wing, chord_vector_wing
                )
            if comp_opt == "ac" or comp_opt == "htp":
                (y_vector_htp, cl_vector_htp) = self.resize_htp_vector(y_vector_htp, cl_vector_htp)

            # Save results to defined path ---------------------------------------------------------
            if self.options["result_folder_path"] != "":
                results = [
                    cl_0_wing,
                    cl_x_wing,
                    cl_alpha_wing,
                    cm_0_wing,
                    y_vector_wing,
                    cl_vector_wing,
                    chord_vector_wing,
                    coef_k_wing,
                    cl_0_htp,
                    cl_aoa_htp,
                    cl_alpha_htp,
                    cl_alpha_htp_isolated,
                    y_vector_htp,
                    cl_vector_htp,
                    coef_k_htp,
                    sref_wing,
                ]
                self.save_results(result_file_path, results)

        # Else retrieved results are used, eventually adapted with new area ratio
        else:
            # Read values from result file ---------------------------------------------------------
            (
                cl_0_wing,
                cl_x_wing,
                cl_alpha_wing,
                cm_0_wing,
                y_vector_wing,
                cl_vector_wing,
                chord_vector_wing,
                coef_k_wing,
                cl_0_htp,
                cl_aoa_htp,
                cl_alpha_htp,
                cl_alpha_htp_isolated,
                y_vector_htp,
                cl_vector_htp,
                coef_k_htp,
            ) = self.read_value_from_data(result_file_path, sref_wing, area_ratio, saved_area_ratio)

        return (
            cl_0_wing,
            cl_x_wing,
            cl_alpha_wing,
            cm_0_wing,
            y_vector_wing,
            cl_vector_wing,
            chord_vector_wing,
            coef_k_wing,
            cl_0_htp,
            cl_aoa_htp,
            cl_alpha_htp,
            cl_alpha_htp_isolated,
            y_vector_htp,
            cl_vector_htp,
            coef_k_htp,
        )

    def compute_wing(
        self,
        inputs,
        altitude: float,
        mach: float,
        aoa_angle: float,
        flaps_angle: Optional[float] = 0.0,
        use_airfoil: Optional[bool] = True,
    ):
        """
        VLM computations for the wing alone.

        @param inputs: inputs parameters defined within FAST-OAD-GA
        @param altitude: altitude for aerodynamic calculation in meters
        @param mach: air speed expressed in mach
        @param aoa_angle: air speed angle of attack with respect to aircraft (degree)
        @param flaps_angle: flaps angle in Deg (default=0.0: i.e. no deflection)
        @param use_airfoil: adds the camberline coordinates of the selected airfoil (default=True)
        @return: wing dictionary including aero parameters as keys: y_vector, cl_vector, cd_vector,
        cm_vector, cl, cdi, cm, coef_e
        """

        # Generate geometries
        self._run(inputs,run_opt="wing")

        # Get inputs
        aspect_ratio = float(inputs["data:geometry:wing:aspect_ratio"])
        meanchord = inputs["data:geometry:wing:MAC:length"]

        y2_wing = float(inputs["data:geometry:wing:root:y"])
        semi_span = float(inputs["data:geometry:wing:span"]) / 2.0
        wing_twist = float(inputs["data:geometry:wing:twist"])

        # Initialization
        x_c = self.wing["x_c"]
        panelchord = self.wing["panel_chord"]
        panelsurf = self.wing["panel_surf"]
        if use_airfoil:
            self.generate_curvature(self.wing, self.options["wing_airfoil_file"])
        self.apply_deflection(inputs, flaps_angle)
        self.generate_twist(self.wing, wing_twist, y2_wing, semi_span)
        panelangle_vect = self.wing["panel_angle_vect"]
        aic = self.wing["aic"]
        aic_inv = np.linalg.inv(aic)
        aic_wake = self.wing["aic_wake"]

        # Compute air speed
        v_inf = max(
            Atmosphere(altitude, altitude_in_feet=False).speed_of_sound * mach, 0.01
        )  # avoid V=0 m/s crashes

        # Calculate all the aerodynamic parameters
        panelsurf_sum = np.sum(panelsurf)
        aoa_angle = np.deg2rad(aoa_angle)
        alpha = np.add(panelangle_vect, aoa_angle)
        gamma = -np.dot(aic_inv, alpha) * v_inf
        c_p = -2 / v_inf * np.divide(gamma, panelchord)
        cl_wing = -np.sum(c_p * panelsurf) / panelsurf_sum
        alphaind = np.dot(aic_wake, gamma) / v_inf
        cdind_panel = c_p * alphaind
        cdi_wing = np.sum(cdind_panel * panelsurf) / panelsurf_sum
        wing_e = cl_wing ** 2 / (np.pi * aspect_ratio * cdi_wing) * 0.955  # !!!: manual correction?
        cmpanel = np.multiply(c_p, (x_c[: self.n_x * self.n_y] - meanchord / 4))
        cm_wing = np.sum(cmpanel * panelsurf) / panelsurf_sum

        # Calculate curves
        wing_cl_vect = []
        wing_y_vect = []
        wing_chord_vect = []
        yc_wing = self.wing["yc"]
        chord_wing = self.wing["chord"]
        for j in range(self.n_y):
            cl_span = 0.0
            y_local = yc_wing[j]
            chord = (chord_wing[j] + chord_wing[j + 1]) / 2.0
            for i in range(self.n_x):
                cl_span += -c_p[i * self.n_y + j] * panelchord[i * self.n_y + j] / chord
            wing_cl_vect.append(cl_span)
            wing_y_vect.append(y_local)
            wing_chord_vect.append(chord)

        # Return values
        wing = {
            "y_vector": wing_y_vect,
            "cl_vector": wing_cl_vect,
            "chord_vector": wing_chord_vect,
            "cd_vector": [],
            "cm_vector": [],
            "cl": cl_wing,
            "cdi": cdi_wing,
            "cm": cm_wing,
            "coef_e": wing_e,
        }

        return wing

    def compute_htp(
        self,
        inputs,
        altitude: float,
        mach: float,
        aoa_angle: float,
        use_airfoil: Optional[bool] = True,
    ):
        """
        VLM computation for the horizontal tail alone.

        @param inputs: inputs parameters defined within FAST-OAD-GA.
        @param altitude: altitude for aerodynamic calculation in meters.
        @param mach: air speed expressed in mach.
        @param aoa_angle: air speed angle of attack with respect to aircraft (degree).
        @param use_airfoil: adds the camberline coordinates of the selected airfoil (default=True).
        @return: htp dictionary including aero parameters as keys: y_vector, cl_vector, cd_vector,
        cm_vector, cl, cdi, cm, coef_e.
        """

        # Generate geometries
        self._run(inputs,run_opt="htp")

        # Get inputs
        aspect_ratio = float(inputs["data:geometry:horizontal_tail:aspect_ratio"])
        meanchord = inputs["data:geometry:horizontal_tail:MAC:length"]

        # Initialization
        x_c = self.htp["x_c"]
        panelchord = self.htp["panel_chord"]
        panelsurf = self.htp["panel_surf"]
        if use_airfoil:
            self.generate_curvature(self.htp, self.options["htp_airfoil_file"])
        panelangle_vect = self.htp["panel_angle_vect"]
        aic = self.htp["aic"]
        aic_inv = np.linalg.inv(aic)
        aic_wake = self.htp["aic_wake"]

        # Compute air speed
        v_inf = max(
            Atmosphere(altitude, altitude_in_feet=False).speed_of_sound * mach, 0.01
        )  # avoid V=0 m/s crashes

        # Calculate all the aerodynamic parameters
        panelsurf_sum = np.sum(panelsurf)
        aoa_angle = np.deg2rad(aoa_angle)
        alpha = np.add(panelangle_vect, aoa_angle) 
        gamma = -np.dot(aic_inv, alpha) * v_inf
        c_p = -2 / v_inf * np.divide(gamma, panelchord)
        cl_htp = -np.sum(c_p * panelsurf) / panelsurf_sum
        alphaind = np.dot(aic_wake, gamma) / v_inf
        cdind_panel = c_p * alphaind
        cdi_htp = np.sum(cdind_panel * panelsurf) / panelsurf_sum
        htp_e = cl_htp ** 2 / (np.pi * aspect_ratio * max(cdi_htp, 1e-12))  # avoid 0.0 division
        cmpanel = np.multiply(c_p, (x_c[: self.n_x * self.n_y] - meanchord / 4))
        cm_htp = np.sum(cmpanel * panelsurf) / panelsurf_sum

        # Calculate curves
        htp_cl_vect = []
        htp_y_vect = []
        yc_htp = self.htp["yc"]
        chord_htp = self.htp["chord"]
        for j in range(self.n_y):
            cl_span = 0.0
            y_local = yc_htp[j]
            chord = (chord_htp[j] + chord_htp[j + 1]) / 2.0
            for i in range(self.n_x):
                cl_span += -c_p[i * self.n_y + j] * panelchord[i * self.n_y + j] / chord
            htp_cl_vect.append(cl_span)
            htp_y_vect.append(y_local)

        # Return values
        htp = {
            "y_vector": htp_y_vect,
            "cl_vector": htp_cl_vect,
            "cd_vector": [],
            "cm_vector": [],
            "cl": cl_htp,
            "cdi": cdi_htp,
            "cm": cm_htp,
            "coef_e": htp_e,
        }

        return htp

    def compute_aircraft(
        self,
        inputs,
        altitude: float,
        mach: float,
        aoa_angle: float,
        flaps_angle: Optional[float] = 0.0,
        use_airfoil: Optional[bool] = True,
    ):
        """
        VLM computation for the complete aircraft.

        @param inputs: inputs parameters defined within FAST-OAD-GA.
        @param altitude: altitude for aerodynamic calculation in meters.
        @param mach: air speed expressed in mach.
        @param aoa_angle: air speed angle of attack with respect to aircraft (degree).
        @param use_airfoil: adds the camberline coordinates of the selected airfoil (default=True).
        @param flaps_angle: flaps angle in Deg (default=0.0: i.e. no deflection).
        @return: wing/htp and aircraft dictionaries including their respective aerodynamic
        coefficients.
        """

        # Get inputs
        aspect_ratio_wing = float(inputs["data:geometry:wing:aspect_ratio"])

        # Compute wing
        wing = self.compute_wing(
            inputs, altitude, mach, aoa_angle, flaps_angle=flaps_angle, use_airfoil=use_airfoil
        )

        # Calculate downwash angle based on Gudmundsson model (p.467)
        cl_wing = wing["cl"]
        beta = np.sqrt(1 - mach ** 2)  # Prandtl-Glauert
        downwash_angle = 2.0 * np.array(cl_wing) / beta * 180.0 / (aspect_ratio_wing * np.pi ** 2)
        aoa_angle_corrected = aoa_angle - downwash_angle

        # Compute htp
        htp = self.compute_htp(inputs, altitude, mach, aoa_angle_corrected, use_airfoil=True)

        # Save results at aircraft level
        aircraft = {"cl": wing["cl"] + htp["cl"], "cd0": None, "cdi": None, "coef_e": None}

        return wing, htp, aircraft

    def _run(self, inputs,run_opt="wing"):

        wing_break = float(inputs["data:geometry:wing:kink:span_ratio"])

        # Define mesh size
        self.n_x = int(DEFAULT_NX)
        if wing_break > 0.0:
            self.ny1 = int(DEFAULT_NY1 + 5)  # n° of panels in the straight section of the wing
            self.ny2 = int(
                (DEFAULT_NY2 - 5) / 2
            )  # n° of panels in in the flapped portion of the wing
        else:
            self.ny1 = int(DEFAULT_NY1)  # n° of panels in the straight section of the wing
            self.ny2 = int(DEFAULT_NY2 / 2)  # n° of panels in in the flapped portion of the wing
        self.ny3 = self.ny2  # n° of panels in the un-flapped exterior portion of the wing

        self.n_y = int(self.ny1 + self.ny2 + self.ny3)

        # Generate WING
        if run_opt == "wing":
            self.wing = {
            "x_panel": np.zeros((self.n_x + 1, 2 * self.n_y + 1)),
            "y_panel": np.zeros(2 * self.n_y + 1),
            "z": np.zeros(self.n_x + 1),
            "x_le": np.zeros(2 * self.n_y + 1),
            "chord": np.zeros(2 * self.n_y + 1),
            "panel_span": np.zeros(2 * self.n_y),
            "panel_chord": np.zeros(self.n_x * self.n_y),
            "panel_surf": np.zeros(self.n_x * self.n_y),
            "x_c": np.zeros(self.n_x * 2 * self.n_y),
            "yc": np.zeros(self.n_x * 2 * self.n_y),
            "x1": np.zeros(self.n_x * 2 * self.n_y),
            "x2": np.zeros(self.n_x * 2 * self.n_y),
            "y1": np.zeros(self.n_x * 2 * self.n_y),
            "y2": np.zeros(self.n_x * 2 * self.n_y),
            "panel_angle": np.zeros(self.n_x),
            "panel_angle_vect": np.zeros(self.n_x * self.n_y),
            "aic": np.zeros((self.n_x * self.n_y, self.n_x * self.n_y)),
            "aic_wake": np.zeros((self.n_x * self.n_y, self.n_x * self.n_y)),
            }
            self._generate_wing(inputs)
        
        # Generate HTP
        if run_opt == "htp":
        # Define elements
            self.htp = {
            "x_panel": np.zeros((self.n_x + 1, 2 * self.n_y + 1)),
            "y_panel": np.zeros(2 * self.n_y + 1),
            "z": np.zeros(self.n_x + 1),
            "x_le": np.zeros(2 * self.n_y + 1),
            "chord": np.zeros(2 * self.n_y + 1),
            "panel_span": np.zeros(2 * self.n_y),
            "panel_chord": np.zeros(self.n_x * self.n_y),
            "panel_surf": np.zeros(self.n_x * self.n_y),
            "x_c": np.zeros(self.n_x * 2 * self.n_y),
            "yc": np.zeros(self.n_x * 2 * self.n_y),
            "x1": np.zeros(self.n_x * 2 * self.n_y),
            "x2": np.zeros(self.n_x * 2 * self.n_y),
            "y1": np.zeros(self.n_x * 2 * self.n_y),
            "y2": np.zeros(self.n_x * 2 * self.n_y),
            "panel_angle": np.zeros(self.n_x),
            "panel_angle_vect": np.zeros(self.n_x * self.n_y),
            "aic": np.zeros((self.n_x * self.n_y, self.n_x * self.n_y)),
            "aic_wake": np.zeros((self.n_x * self.n_y, self.n_x * self.n_y)),
        }
            self._generate_htp(inputs)

    def _generate_wing(self, inputs):
        """Generates the coordinates for VLM calculations and aic matrix of the wing.
             Pi +......> y     Given a trapezoid defined by vertices Pi and Pf
                | \            and chords 1 and 2 representing a wing segment
                |  \           that complies with the VLM theory, returns the
                |   + Pf       points and panels of the mesh:
          chord1|   |
                |   |             - Points are given as a list of Np elements,
                |   |chord2         being Np the number of points of the mesh.
                +---+
                |                 - Panels are given as a list of list of NP
                x 				      elements, each element composed of 4 points,
                                    where NP is the number of panels of the mesh.
                x - chordwise
                    direction    The corner points of each panel are arranged in a
                y - spanwise     clockwise fashion following this order:
                    direction
            x_le array identifies the x-coordinate of Pi and Pf along the span-direction
        
        """
        y2_wing = inputs["data:geometry:wing:root:y"]
        semi_span = inputs["data:geometry:wing:span"] / 2.0
        root_chord = inputs["data:geometry:wing:root:chord"]
        tip_chord = inputs["data:geometry:wing:tip:chord"]
        flap_span_ratio = inputs["data:geometry:flap:span_ratio"]

        # Initial data (zero matrix/array)
        y_panel = self.wing["y_panel"]
        x_le = self.wing["x_le"]
        y_endflaps = y2_wing + flap_span_ratio * (semi_span - y2_wing)
        # Definition of x_panel, y_panel, x_le and chord (Right side)

        indices = np.indices(y_panel.shape)[0].astype(float)

        y_panel = y2_wing * indices / self.ny1

        y_panel = np.where(
            indices >= self.ny1,
            y2_wing + (y_endflaps - y2_wing) * (indices - self.ny1) / self.ny2,
            y_panel,
        )
        y_tapered_section = y_panel - y2_wing
        chord = np.where(
            indices >= self.ny1,
            root_chord + (tip_chord - root_chord) * y_tapered_section / (semi_span - y2_wing),
            np.full_like(indices, root_chord),
        )
        x_le = np.where(
            indices >= self.ny1,
            y_tapered_section * (root_chord - tip_chord) / (4 * (semi_span - y2_wing)),
            x_le,
        )

        y_panel = np.where(
            indices >= self.ny1 + self.ny2,
            y_endflaps + (semi_span - y_endflaps) * (indices - (self.ny1 + self.ny2)) / self.ny3,
            y_panel,
        )
        y_tapered_section = y_panel - y2_wing
        chord = np.where(
            indices >= self.ny1 + self.ny2,
            root_chord + (tip_chord - root_chord) * y_tapered_section / (semi_span - y2_wing),
            chord,
        )
        x_le = np.where(
            indices >= self.ny1 + self.ny2,
            y_tapered_section * (root_chord - tip_chord) / (4 * (semi_span - y2_wing)),
            x_le,
        )

        # Mirror the right side to define the left side for y_panel, x_le and chord
        y_panel[self.n_y + 1 :] = -y_panel[1 : self.n_y + 1]
        chord[self.n_y + 1 :] = chord[1 : self.n_y + 1]
        x_le[self.n_y + 1 :] = x_le[1 : self.n_y + 1]
        # Save data
        self.wing["y_panel"] = y_panel
        self.wing["chord"] = chord
        self.wing["x_le"] = x_le
        # Launch common code
        self._generate_common(self.wing)

    def _generate_htp(self, inputs):
        """Generates the coordinates for VLM calculations and AIC matrix of the htp.
             Pi +......> y     Given a trapezoid defined by vertices Pi and Pf
                | \            and chords 1 and 2 representing a wing segment
                |  \           that complies with the VLM theory, returns the
                |   + Pf       points and panels of the mesh:
          chord1|   |
                |   |             - Points are given as a list of Np elements,
                |   |chord2         being Np the number of points of the mesh.
                +---+
                |                 - Panels are given as a list of list of NP
                x 				      elements, each element composed of 4 points,
                                    where NP is the number of panels of the mesh.
                x - chordwise
                    direction    The corner points of each panel are arranged in a
                y - spanwise     clockwise fashion following this order:
                    direction
            x_le array identifies the x-coordinate of Pi and Pf along the span-direction

   
        """
        semi_span = inputs["data:geometry:horizontal_tail:span"] / 2.0
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]

        # Initial data (zero matrix/array)
        y_panel = self.htp["y_panel"]
        # Definition of x_panel, y_panel, x_le and chord (Right side)

        indices = np.indices(y_panel.shape)[0].astype(float)
        y_panel = semi_span * indices / self.n_y
        chord = root_chord + (tip_chord - root_chord) * y_panel / semi_span
        x_le = y_panel * (root_chord - tip_chord) / (4 * semi_span)

        # Mirror the right side to define the left side for  y_panel, x_le and chord
        y_panel[self.n_y + 1 :] = -y_panel[1 : self.n_y + 1]
        chord[self.n_y + 1 :] = chord[1 : self.n_y + 1]
        x_le[self.n_y + 1 :] = x_le[1 : self.n_y + 1]

        # Save data
        self.htp["y_panel"] = y_panel
        self.htp["chord"] = chord
        self.htp["x_le"] = x_le
        # Launch common code
        self._generate_common(self.htp)

    def _generate_common(self, dictionary):
        """Common code shared between wing and htp to calculate geometry/aero parameters.

                ^
              y |                Points defining the panel
                |                are named clockwise. A(x_1,y_1), B(x_2,y_2), P(x_c,y_c)
        P3--B---|-----P4         Reference:
        |   |   |     |          John J. Bertin, Russell M. Cummings - Aerodynamics for Engineers
        |   |   |     |          p.394-398
        T1  |   +--P--T2---->
        |   |         |     x
        |   |         |
        P2--A--------P1

        """
        # Initial data (zero matrix/array)
        x_le = dictionary["x_le"]
        chord = dictionary["chord"]
        x_panel = dictionary["x_panel"]
        y_panel = dictionary["y_panel"]
        panelchord = dictionary["panel_chord"]
        panelsurf = dictionary["panel_surf"]
        x_c = dictionary["x_c"]
        y_c = dictionary["yc"]
        x_1 = dictionary["x1"]
        y_1 = dictionary["y1"]
        x_2 = dictionary["x2"]
        y_2 = dictionary["y2"]
        aic = dictionary["aic"]
        aic_wake = dictionary["aic_wake"]

        n_x = self.n_x
        n_y = self.n_y
        # calculate panel geometry and coordinate value
        (
            x_panel,
            y_panel,
            x_le,
            chord,
            panelchord,
            panelspan,
            panelsurf,
            x_1,
            y_1,
            x_2,
            y_2,
            x_c,
            y_c,
        ) = self.panle_point_calculation(
            x_panel,
            y_panel,
            x_le,
            chord,
            panelchord,
            panelsurf,
            x_1,
            y_1,
            x_2,
            y_2,
            x_c,
            y_c,
            n_x,
            n_y,
        )

        # Aerodynamic coefficients computation
        aic, aic_wake = self.aic_computation(x_1, y_1, x_2, y_2, x_c, y_c, aic, aic_wake, n_x, n_y)
        # Save data
        dictionary["x_panel"] = x_panel
        dictionary["panel_span"] = panelspan
        dictionary["panel_chord"] = panelchord
        dictionary["panel_surf"] = panelsurf
        dictionary["x_c"] = x_c
        dictionary["yc"] = y_c
        dictionary["x1"] = x_1
        dictionary["y1"] = y_1
        dictionary["x2"] = x_2
        dictionary["y2"] = y_2
        dictionary["aic"] = aic
        dictionary["aic_wake"] = aic_wake

    def generate_twist(self, dictionary, twist, y_start, y_end):
        """
        Add the twist on the lifting surface assuming a linear variation between y_start and y_end.

        @param dictionary: dictionary which contains the point coordinates of the lifting surface
        @param twist: wing twist variation between root and tip, <0 when tip is pointed downwards
        @param y_start: y coordinate of the start of linear twist
        @param y_end:y coordinate of the end of linear twist
        """

        y_panel = dictionary["y_panel"][: self.n_y]
        twist = np.interp(y_panel, [y_start, y_end], [0.0, twist])
        twist_tile = np.tile(twist, self.n_x)

        panel_angle_vect = dictionary["panel_angle_vect"]

        dictionary["panel_angle_vect"] = panel_angle_vect + twist_tile

    def generate_curvature(self, dictionary, file_name):
        """Generates curvature corresponding to the airfoil contained in .af file."""
        x_panel = dictionary["x_panel"]

        # Initialization
        z_panel = np.zeros(self.n_x + 1)
        rootchord = x_panel[self.n_x, 0] - x_panel[0, 0]
        # Calculation of panelangle_vect
        profile = get_profile(
            airfoil_folder_path=self.options["airfoil_folder_path"], file_name=file_name
        )
        mean_line = profile.get_mean_line()
        for i in range(self.n_x + 1):
            xred = (x_panel[i, 0] - x_panel[0, 0]) / rootchord
            z_panel[i] = np.interp(
                np.clip(xred, min(mean_line["x"]), max(mean_line["x"])),
                mean_line["x"],
                mean_line["z"],
            )
        z_panel = z_panel * rootchord
        x_coord_chord = x_panel[:, 0]

        panelangle = (z_panel[: self.n_x] - z_panel[1 : self.n_x + 1]) / (
            x_coord_chord[1 : self.n_x + 1] - x_coord_chord[: self.n_x]
        )
        panelangle_vect = np.repeat(panelangle, self.n_y)

        # Save results
        dictionary["panel_angle_vect"] = panelangle_vect
        dictionary["panel_angle"] = panelangle
        dictionary["z"] = z_panel

    def apply_deflection(self, inputs, deflection_angle):
        """Apply panel angle deflection due to flaps angle [UNUSED: deflection_angle=0.0]."""

        root_chord = inputs["data:geometry:wing:root:chord"]
        x_start = (1.0 - inputs["data:geometry:flap:chord_ratio"]) * root_chord

        deflection_angle *= np.pi / 180  # converted to radian
        x_panel = self.wing["x_panel"]
        panelangle = self.wing["panel_angle"]
        panelangle_vect = self.wing["panel_angle_vect"]

        z_panel = np.zeros(self.n_x + 1)
        z_panel_no_flaps = copy.deepcopy(self.wing["z"])
        for i in range(self.n_x + 1):
            if x_panel[i, 0] > x_start:
                z_panel[i] = z_panel_no_flaps[i] - np.sin(deflection_angle) * (
                    x_panel[i, 0] - x_start
                )
            else:
                z_panel[i] = z_panel_no_flaps[i]

        # At this point z_panel contains the z coordinate of the flapped part of the wing,
        # panel angle vector should not change in the un-flapped part
        for i in range(self.n_x):
            panelangle[i] = (z_panel[i] - z_panel[i + 1]) / (x_panel[i + 1, 0] - x_panel[i, 0])

        # At this point panelangle contains the panel angle of the flapped part of the wing,
        # panel angle vector should not change in the un-flapped part

        # If we are in the flapped portion of the wing
        for j in range(self.ny1, self.ny1 + self.ny2):
            for i in range(self.n_x):
                panelangle_vect[i * self.n_y + j] = panelangle[i]

        # Save results
        self.wing["panel_angle_vect"] = panelangle_vect
        self.wing["panel_angle"] = panelangle
        self.wing["z"] = z_panel

    @staticmethod
    def _interpolate_cdp(lift_coeff: np.ndarray, drag_coeff: np.ndarray, ojective: float) -> float:
        """

        :param lift_coeff: CL array
        :param drag_coeff: CDp array
        :param ojective: CL_ref objective value
        :return: CD_ref if CL_ref encountered, or default value otherwise.
        """
        # Reduce vectors for interpolation
        for idx in range(len(lift_coeff)):
            if np.sum(lift_coeff[idx : len(lift_coeff)] == 0) == (len(lift_coeff) - idx):
                lift_coeff = lift_coeff[0:idx]
                drag_coeff = drag_coeff[0:idx]
                break

        # Interpolate value if within the interpolation range
        if min(lift_coeff) <= ojective <= max(lift_coeff):
            idx_max = int(float(np.where(lift_coeff == max(lift_coeff))[0]))
            return np.interp(ojective, lift_coeff[0 : idx_max + 1], drag_coeff[0 : idx_max + 1])
        elif ojective < lift_coeff[0]:
            cdp = drag_coeff[0] + (ojective - lift_coeff[0]) * (drag_coeff[1] - drag_coeff[0]) / (
                lift_coeff[1] - lift_coeff[0]
            )
        else:
            cdp = drag_coeff[-1] + (ojective - lift_coeff[-1]) * (
                drag_coeff[-1] - drag_coeff[-2]
            ) / (lift_coeff[-1] - lift_coeff[-2])
        _LOGGER.warning("CL not in range. Linear extrapolation of CDp value %f", cdp)
        return cdp

    @staticmethod
    def search_results(result_folder_path, geometry_set):
        """Search the results folder to see if the geometry has already been calculated."""
        # default value
        result_file_path = None
        saved_area_ratio = 1.0
        if os.path.exists(result_folder_path):
            geometry_set_labels = [
                "sweep25_wing",
                "taper_ratio_wing",
                "aspect_ratio_wing",
                "dihedral_angle_wing",
                "twist_angle_wing",
                "sweep25_htp",
                "taper_ratio_htp",
                "aspect_ratio_htp",
                "mach",
                "area_ratio",
            ]
            # If some results already stored search for corresponding geometry
            if pth.exists(pth.join(result_folder_path, "geometry_0.csv")):
                idx = 0
                while pth.exists(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv")):
                    if pth.exists(pth.join(result_folder_path, "vlm_" + str(idx) + ".csv")):
                        data = pd.read_csv(
                            pth.join(result_folder_path, "geometry_" + str(idx) + ".csv")
                        )
                        values = data.to_numpy()[:, 1].tolist()
                        labels = data.to_numpy()[:, 0].tolist()
                        data = pd.DataFrame(values, index=labels)
                        # noinspection PyBroadException
                        try:
                            if (
                                np.size(data.loc[geometry_set_labels[0:-1], 0].to_numpy())
                                == len(geometry_set_labels) - 1
                            ):
                                saved_set = np.around(
                                    data.loc[geometry_set_labels[0:-1], 0].to_numpy(), decimals=6
                                )
                                if (
                                    np.sum(saved_set == geometry_set[0:-1])
                                    == len(geometry_set_labels) - 1
                                ):
                                    result_file_path = pth.join(
                                        result_folder_path, "vlm_" + str(idx) + ".csv"
                                    )
                                    saved_area_ratio = data.loc["area_ratio", 0]
                                    return result_file_path, saved_area_ratio
                        except Exception:
                            break
                    idx += 1

        return result_file_path, saved_area_ratio

    @staticmethod
    def save_geometry(result_folder_path, geometry_set):
        """Save geometry if not already computed by finding first available index."""
        geometry_set_labels = [
            "sweep25_wing",
            "taper_ratio_wing",
            "aspect_ratio_wing",
            "dihedral_angle_wing",
            "twist_angle_wing",
            "sweep25_htp",
            "taper_ratio_htp",
            "aspect_ratio_htp",
            "mach",
            "area_ratio",
        ]
        data = pd.DataFrame(geometry_set, index=geometry_set_labels)
        idx = 0
        while pth.exists(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv")):
            idx += 1
        data.to_csv(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv"))
        result_file_path = pth.join(result_folder_path, "vlm_" + str(idx) + ".csv")

        return result_file_path

    @staticmethod
    def save_results(result_file_path, results):
        """Reads saved results."""
        labels = [
            "cl_0_wing",
            "cl_X_wing",
            "cl_alpha_wing",
            "cm_0_wing",
            "y_vector_wing",
            "cl_vector_wing",
            "chord_vector_wing",
            "coef_k_wing",
            "cl_0_htp",
            "cl_X_htp",
            "cl_alpha_htp",
            "cl_alpha_htp_isolated",
            "y_vector_htp",
            "cl_vector_htp",
            "coef_k_htp",
            "saved_ref_area",
        ]
        data = pd.DataFrame(results, index=labels)
        data.to_csv(result_file_path)

    @staticmethod
    def read_results(result_file_path):

        data = pd.read_csv(result_file_path)
        values = data.to_numpy()[:, 1].tolist()
        labels = data.to_numpy()[:, 0].tolist()

        return pd.DataFrame(values, index=labels)

    def post_processing_wing(
        self,
        width_max,
        span_wing,
        mach,
        dihedral_angle,
        wing_0,
        wing_aoa,
        aoa_angle,
        aspect_ratio_wing,
        cl_wing_airfoil,
        cdp_wing_airfoil,
    ):
        """_summary_
        Calculate aerodynamic characteristic with VLM results
        Args:
            width_max (_float_): maximun wi
            span_wing (_float_): _description_
            mach (_float_): mach number
            dihedral_angle (_float_): wing dihedral angle
            wing_0 (_float_): wing aerodynamic characteristics with zero angle of attack
            wing_aoa (_float_): wing aerodynamic characteristics with specific angle of attack
            aoa_angle (_float_): angle of attakc data list
            aspect_ratio_wing (_float_): wing aspect ratio
            cl_wing_airfoil (_list_): lift coefficient data of wing respect to different angle of attack
            cdp_wing_airfoil (_list_): pressure drag coefficient data of wing respect to different angle of attack

        Returns:
            _list_: post-processed data for other use
        """
        k_fus = 1 + 0.025 * width_max / span_wing - 0.025 * (width_max / span_wing) ** 2
        beta = np.sqrt(1 - mach ** 2)  # Prandtl-Glauert
        cl_0_wing = float((wing_0["cl"] * k_fus / beta) * np.cos(dihedral_angle) ** 2.0)
        cl_x_wing = float(wing_aoa["cl"] * k_fus / beta)
        cm_0_wing = float(wing_0["cm"] * k_fus / beta)
        cl_alpha_wing = ((cl_x_wing - cl_0_wing) / (aoa_angle * np.pi / 180)) * np.cos(
            dihedral_angle
        ) ** 2.0

        # The correction on the effect of dihedral angle cannot be taken into account by
        # modifying the generation of panel since only the surfaces of the panel and their
        # aoa impact the computation, so we will stick with this "post processing" of the
        # dihedral

        y_vector_wing = wing_aoa["y_vector"]
        cl_vector_wing = (np.array(wing_aoa["cl_vector"]) * k_fus / beta).tolist()
        chord_vector_wing = wing_aoa["chord_vector"]
        cdp_foil = self._interpolate_cdp(cl_wing_airfoil, cdp_wing_airfoil, cl_x_wing)
        # Mach correction
        if mach <= 0.4:
            coef_e = wing_aoa["coef_e"]
        else:
            coef_e = wing_aoa["coef_e"] * (
                -0.001521 * ((mach - 0.05) / 0.3 - 1) ** 10.82 + 1
            )  
        cdi = cl_x_wing ** 2 / (np.pi * aspect_ratio_wing * coef_e) + cdp_foil
        coef_e = wing_aoa["cl"] ** 2 / (np.pi * aspect_ratio_wing * cdi)
        # Fuselage correction
        k_fus = 1 - 2 * (width_max / span_wing) ** 2  
        coef_e = float(coef_e * k_fus)
        coef_k_wing = float(1.0 / (np.pi * aspect_ratio_wing * coef_e))

        return (
            beta,
            cl_0_wing,
            cl_x_wing,
            cm_0_wing,
            cl_alpha_wing,
            y_vector_wing,
            cl_vector_wing,
            chord_vector_wing,
            coef_k_wing,
        )

    def post_processing_htp_ac(
        self,
        beta,
        aspect_ratio_htp,
        area_ratio,
        mach,
        aoa_angle,
        cl_htp_airfoil,
        cdp_htp_airfoil,
        htp_0,
        htp_aoa,
    ):
        """_summary_
        Calculate aerodynamic characteristic with VLM results
        Args:
            beta (_float_): sweep angle of horizontal stabilizer
            aspect_ratio_htp (_float_): wing aspect ratio
            area_ratio (_float_): area ratio between wing and horizontal stabilizer
            mach (_float_): mack number
            aoa_angle (_list_): angle of attakc data list
            cl_htp_airfoil (_list_): lift coefficient data of horizontal stabilizer respect to different angle of attack
            cdp_htp_airfoil (_list_): pressure drag coefficient data of horizontal stabilizer respect to different angle of attack
            htp_0 (_list_): horizontal stabilizer aerodynamic charateristic data list
                            with zero angle of attack
            htp_aoa (_list_): horizontal stabilizer aerodynamic charateristic data list
                              with specified angle of attack

        Returns:
            _list_: post-processed data for other use
        """
        cl_0_htp = float(htp_0["cl"]) / beta * area_ratio
        cl_aoa_htp = float(htp_aoa["cl"]) / beta * area_ratio
        cl_alpha_htp = float((cl_aoa_htp - cl_0_htp) / (aoa_angle * np.pi / 180))
        cdp_foil = self._interpolate_cdp(cl_htp_airfoil, cdp_htp_airfoil, htp_aoa["cl"] / beta)
        # Mach correction
        if mach <= 0.4:
            coef_e = htp_aoa["coef_e"]
        else:
            coef_e = htp_aoa["coef_e"] * (
                -0.001521 * ((mach - 0.05) / 0.3 - 1) ** 10.82 + 1
            )  
        cdi = (htp_aoa["cl"] / beta) ** 2 / (np.pi * aspect_ratio_htp * coef_e) + cdp_foil
        coef_k_htp = float(cdi / cl_aoa_htp ** 2 * area_ratio)
        y_vector_htp = htp_aoa["y_vector"]
        cl_vector_htp = (np.array(htp_aoa["cl_vector"]) / beta * area_ratio).tolist()

        return (
            cl_0_htp,
            cl_aoa_htp,
            cl_alpha_htp,
            coef_k_htp,
            y_vector_htp,
            cl_vector_htp,
        )

    def read_value_from_data(self, result_file_path, sref_wing, area_ratio, saved_area_ratio):
        """_summary_
        Read existed data to speed up the process
        Args:
            result_file_path (_path_): computation result path
            sref_wing (_float_): wing reference area
            area_ratio (_float_): area ratio between wing and horizontal stabilizer
            saved_area_ratio (_float_):

        Returns:
            _type_: _description_
        """
        data = self.read_results(result_file_path)
        saved_area_wing = float(data.loc["saved_ref_area", 0])
        cl_0_wing = float(data.loc["cl_0_wing", 0])
        cl_x_wing = float(data.loc["cl_X_wing", 0])
        cl_alpha_wing = float(data.loc["cl_alpha_wing", 0])
        cm_0_wing = float(data.loc["cm_0_wing", 0])
        y_vector_wing = np.array(
            [float(i) for i in data.loc["y_vector_wing", 0][1:-2].split(",")]
        ) * np.sqrt(sref_wing / saved_area_wing)
        cl_vector_wing = np.array(
            [float(i) for i in data.loc["cl_vector_wing", 0][1:-2].split(",")]
        )
        chord_vector_wing = np.array(
            [float(i) for i in data.loc["chord_vector_wing", 0][1:-2].split(",")]
        ) * np.sqrt(sref_wing / saved_area_wing)
        coef_k_wing = float(data.loc["coef_k_wing", 0])
        cl_0_htp = float(data.loc["cl_0_htp", 0]) * (area_ratio / saved_area_ratio)
        cl_aoa_htp = float(data.loc["cl_X_htp", 0]) * (area_ratio / saved_area_ratio)
        cl_alpha_htp = float(data.loc["cl_alpha_htp", 0]) * (area_ratio / saved_area_ratio)
        cl_alpha_htp_isolated = float(data.loc["cl_alpha_htp_isolated", 0]) * (
            area_ratio / saved_area_ratio
        )
        y_vector_htp = np.array([float(i) for i in data.loc["y_vector_htp", 0][1:-2].split(",")])
        cl_vector_htp = np.array([float(i) for i in data.loc["cl_vector_htp", 0][1:-2].split(",")])
        coef_k_htp = float(data.loc["coef_k_htp", 0]) * (area_ratio / saved_area_ratio)

        return (
            cl_0_wing,
            cl_x_wing,
            cl_alpha_wing,
            cm_0_wing,
            y_vector_wing,
            cl_vector_wing,
            chord_vector_wing,
            coef_k_wing,
            cl_0_htp,
            cl_aoa_htp,
            cl_alpha_htp,
            cl_alpha_htp_isolated,
            y_vector_htp,
            cl_vector_htp,
            coef_k_htp,
        )

    def resize_wing_vector(self, y_vector_wing, cl_vector_wing, chord_vector_wing):
        """_summary_
        Resize wing vector for later use in other model
        Args:
            y_vector_wing (_list_): wing position data
            cl_vector_wing (_list_): wing lift coefficient data
            chord_vector_wing (_list_): wing chord length data

        Returns:
            _list_: reformed wing data
        """
        if SPAN_MESH_POINT < len(y_vector_wing):
            y_interp = np.linspace(y_vector_wing[0], y_vector_wing[-1], SPAN_MESH_POINT)
            cl_vector_wing = np.interp(y_interp, y_vector_wing, cl_vector_wing)
            chord_vector_wing = np.interp(y_interp, y_vector_wing, chord_vector_wing)
            y_vector_wing = y_interp
            warnings.warn("Defined maximum span mesh in fast aerodynamics\\constants.py exceeded!")
        else:
            additional_zeros = list(np.zeros(SPAN_MESH_POINT - len(y_vector_wing)))
            y_vector_wing.extend(additional_zeros)
            cl_vector_wing.extend(additional_zeros)
            chord_vector_wing.extend(additional_zeros)
        return (y_vector_wing, cl_vector_wing, chord_vector_wing)

    def resize_htp_vector(self, y_vector_htp, cl_vector_htp):
        """_summary_
        Resize horizontal stabilizer vector for later use in other model
        Args:
            y_vector_htp (_list_): horizontal stabilizer poistion data
            cl_vector_htp (_list_): horizontal lift coefficient data

        Returns:
            _list_: reform horizontal stabilizer data
        """
        if SPAN_MESH_POINT < len(y_vector_htp):
            y_interp = np.linspace(y_vector_htp[0], y_vector_htp[-1], SPAN_MESH_POINT)
            cl_vector_htp = np.interp(y_interp, y_vector_htp, cl_vector_htp)
            y_vector_htp = y_interp
            warnings.warn("Defined maximum span mesh in fast aerodynamics\\constants.py exceeded!")
        else:
            additional_zeros = list(np.zeros(SPAN_MESH_POINT - len(y_vector_htp)))
            y_vector_htp.extend(additional_zeros)
            cl_vector_htp.extend(additional_zeros)
        return (y_vector_htp, cl_vector_htp)
    
    def aic_computation(self, x_1, y_1, x_2, y_2, x_c, y_c, aic, aic_wake, n_x, n_y):
        """
        
                ^
              y |                Points defining the panel
                |                are named clockwise. A(x_1,y_1), B(x_2,y_2), P(x_c,y_c)
        P3--B---|-----P4         Reference:
        |   |   |     |          John J. Bertin, Russell M. Cummings - Aerodynamics for Engineers
        |   |   |     |          p.394-398
        T1  |   +--P--T2---->       
        |   |         |     x
        |   |         |
        P2--A--------P1
        
        #The P point is the center point of all the panel, in AIC matrix construction. For each linear 
         equations in the AIC linear system, each P point of each panel needs is oconsidered (double for loop).  
        
        coeff_9 =  
        (x_2 - x_1) * (x_c - x_1) + (y_2 - y_1) * (y_c - y_1)
        -----------------------------------------------------  
            sqrt[(x_c - x_1)** 2 + (y_c - y_1) ** 2]

        - 

        (x_2 - x_1) * (x_c - x_2) + (y_2 - y_1) * (y_c - y_2)
        -----------------------------------------------------
            sqrt[(x_c - x_2)** 2 + (y_c - y_2) ** 2]


        coeff_10 =
        {1 + (x_c - x_2) / sqrt[(x_c - x_2)** 2 + (y_c - y_2) ** 2]} 
        ------------------------------------------------------------
                                (y_c - y_2)

        -

        {1 + (x_c - x_1) / sqrt[(x_c - x_1)** 2 + (y_c - y_1) ** 2]}
        ------------------------------------------------------------ 
                                (y_2 - y_1)
        
        
        """
        for i in range(n_x * n_y):
            for j in range(n_x * n_y):
                # Right wing
                coeff_1_r = x_c[i] - x_1[j]
                coeff_2_r = y_c[i] - y_1[j]
                coeff_3_r = x_c[i] - x_2[j]
                coeff_4_r = y_c[i] - y_2[j]
                coeff_5_r = np.sqrt(coeff_1_r ** 2 + coeff_2_r ** 2)
                coeff_6_r = np.sqrt(coeff_3_r ** 2 + coeff_4_r ** 2)
                coeff_7_r = x_2[j] - x_1[j]
                coeff_8_r = y_2[j] - y_1[j]
                coeff_9_r = (coeff_7_r * coeff_1_r + coeff_8_r * coeff_2_r) / coeff_5_r - (
                    coeff_7_r * coeff_3_r + coeff_8_r * coeff_4_r
                ) / coeff_6_r
                coeff_10_r = (1 + coeff_3_r / coeff_6_r) / coeff_4_r - (
                    1 + coeff_1_r / coeff_5_r
                ) / coeff_2_r
                #cross(outer) product between vector <CA> and vector vector <CB>
                if coeff_1_r * coeff_4_r - coeff_2_r * coeff_3_r != 0:
                    aic[i, j] = (coeff_9_r / (coeff_1_r * coeff_4_r - coeff_2_r * coeff_3_r)) / (
                        4 * np.pi
                    )
                aic_wake[i, j] = coeff_10_r / (4 * np.pi)
                aic[i, j] = aic[i, j] + coeff_10_r / (4 * np.pi)

                
                # Left wing
                coeff_1_l = x_c[i] - x_1[self.n_x * self.n_y + j]
                coeff_2_l = y_c[i] - y_1[self.n_x * self.n_y + j]
                coeff_3_l = x_c[i] - x_2[self.n_x * self.n_y + j]
                coeff_4_l = y_c[i] - y_2[self.n_x * self.n_y + j]
                coeff_5_l = np.sqrt(coeff_1_l ** 2 + coeff_2_l ** 2)
                coeff_6_l = np.sqrt(coeff_3_l ** 2 + coeff_4_l ** 2)
                coeff_7_l = x_2[self.n_x * self.n_y + j] - x_1[self.n_x * self.n_y + j]
                coeff_8_l = y_2[self.n_x * self.n_y + j] - y_1[self.n_x * self.n_y + j]
                coeff_9_l = (coeff_7_l * coeff_1_l + coeff_8_l * coeff_2_l) / coeff_5_l - (
                    coeff_7_l * coeff_3_l + coeff_8_l * coeff_4_l
                ) / coeff_6_l
                coeff_10_l = (1 + coeff_3_l / coeff_6_l) / coeff_4_l - (
                    1 + coeff_1_l / coeff_5_l
                ) / coeff_2_l
                #cross(outer) product between vector <CA> and vector vector <CB>
                if coeff_1_l * coeff_4_l - coeff_2_l * coeff_3_l != 0:
                    aic[i, j] = aic[i, j] + (
                        coeff_9_l / (coeff_1_l * coeff_4_l - coeff_2_l * coeff_3_l)
                    ) / (4 * np.pi)
                aic_wake[i, j] = aic_wake[i, j] + coeff_10_l / (4 * np.pi)
                aic[i, j] = aic[i, j] + coeff_10_l / (4 * np.pi)
                
        return aic, aic_wake
    
    def panle_point_calculation(
        self,
        x_panel,
        y_panel,
        x_le,
        chord,
        panelchord,
        panelsurf,
        x_1,
        y_1,
        x_2,
        y_2,
        x_c,
        y_c,
        n_x,
        n_y,
    ):

        """
        Calculate the coordinate value of each geometry points in each panel.
        Store this value into position arrays for later AIC matrix computation.
        """
        for i in range(n_x + 1):
            x_panel[i, :] = x_le + chord * i / n_x

        # Calculate panel span with symmetry
        panelspan = y_panel[1:] - y_panel[:-1]
        panelspan[n_y:] = panelspan[:n_y]

        # Calculate characteristic points (Right and left side)
        for i in range(n_x):
            panelchord[i * n_y : (i + 1) * n_y] = 0.5 * (
                (x_panel[i + 1, :n_y] - x_panel[i, :n_y])
                + (x_panel[i + 1, 1 : n_y + 1] - x_panel[i, 1 : n_y + 1])
            )
            panelsurf[i * n_y : (i + 1) * n_y] = (
                panelspan[:n_y] * panelchord[i * n_y : (i + 1) * n_y]
            )
            # right side

            # point A for the right-half
            x_1[i * n_y : (i + 1) * n_y] = x_panel[i, :n_y] + 0.25 * (
                x_panel[i + 1, :n_y] - x_panel[i, :n_y]
            )

            y_1[i * n_y : (i + 1) * n_y] = y_panel[:n_y]

            # point B for the right-half
            x_2[i * n_y : (i + 1) * n_y] = x_panel[i, 1 : n_y + 1] + 0.25 * (
                x_panel[i + 1, 1 : n_y + 1] - x_panel[i, 1 : n_y + 1]
            )

            y_2[i * n_y : (i + 1) * n_y] = y_panel[1 : n_y + 1]

            # point P for the right-half
            x_c[i * n_y : (i + 1) * n_y] = (
                x_panel[i, :n_y] + x_panel[i, 1 : n_y + 1]
            ) * 0.5 + 0.75 * panelchord[i * n_y : (i + 1) * n_y]

            y_c[i * n_y : (i + 1) * n_y] = (y_panel[:n_y] + y_panel[1 : n_y + 1]) * 0.5

            # letf side

            # point A for the left-half
            x_1[n_x * n_y + i * n_y : n_x * n_y + (i + 1) * n_y] = x_panel[
                i, n_y + 1 : 2 * n_y + 1
            ] + 0.25 * (x_panel[i + 1, n_y + 1 : 2 * n_y + 1] - x_panel[i, n_y + 1 : 2 * n_y + 1])

            y_1[n_x * n_y + i * n_y : n_x * n_y + (i + 1) * n_y] = y_panel[n_y + 1 : 2 * n_y + 1]

            # point B for the left-half
            x_2[n_x * n_y + i * n_y : n_x * n_y + (i + 1) * n_y] = np.concatenate(
                (
                    np.array([x_panel[i, 0] + 0.25 * (x_panel[i + 1, 0] - x_panel[i, 0])]),
                    (
                        x_panel[i, n_y : 2 * n_y]
                        + 0.25 * (x_panel[i + 1, n_y : 2 * n_y] - x_panel[i, n_y : 2 * n_y])
                    )[1:],
                )
            )
            y_2[n_x * n_y + i * n_y : n_x * n_y + (i + 1) * n_y] = np.concatenate(
                (np.array([0.0]), y_panel[n_y + 1 : 2 * n_y],)
            )
        # Calculate remaining characteristic points (Left side)
        # point P for the left-half
        x_c[n_x * n_y :] = x_c[: n_x * n_y]
        y_c[n_x * n_y :] = -y_c[: n_x * n_y]

        return (x_panel, y_panel, x_le, chord, panelchord,panelspan, panelsurf, x_1, y_1, x_2, y_2, x_c, y_c)

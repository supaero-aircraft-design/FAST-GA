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

import os

import numpy as np
import openmdao.api as om
import pandas as pd
from stdatm import Atmosphere

from fastga.models.performances.mission.mission_components import (
    POINTS_NB_CRUISE,
    POINTS_NB_CLIMB,
    POINTS_NB_DESCENT,
)

CSV_DATA_LABELS = [
    "time",
    "altitude",
    "ground_distance",
    "mass",
    "x_cg",
    "true_airspeed",
    "equivalent_airspeed",
    "mach",
    "d_vx_dt",
    "density",
    "gamma",
    "alpha",
    "delta_m",
    "cl_wing",
    "cl_htp",
    "cl_aircraft",
    "cd_aircraft",
    "delta_Cl",
    "delta_Cd",
    "delta_Cm",
    "thrust (N)",
    "thrust_rate",
    "engine_setting",
    "tsfc (kg/s/N)",
    "fuel_flow (kg/s)",
    "energy_consumed (W*h)",
    "time step (s)",
    "name",
]


class ToCSV(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be " "treated"
        )
        self.options.declare("out_file", default="", types=str)

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "d_vx_dt", shape=number_of_points, val=np.full(number_of_points, np.nan), units="m/s**2"
        )
        self.add_input(
            "time", shape=number_of_points, val=np.full(number_of_points, np.nan), units="s"
        )
        self.add_input(
            "x_cg", shape=number_of_points, val=np.full(number_of_points, np.nan), units="m"
        )
        self.add_input(
            "altitude", shape=number_of_points, val=np.full(number_of_points, np.nan), units="m"
        )
        self.add_input(
            "position", shape=number_of_points, val=np.full(number_of_points, np.nan), units="m"
        )
        self.add_input(
            "mass", val=np.full(number_of_points, np.nan), shape=number_of_points, units="kg"
        )
        self.add_input(
            "true_airspeed",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            units="m/s",
        )
        self.add_input(
            "equivalent_airspeed",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            units="m/s",
        )
        self.add_input(
            "gamma", val=np.full(number_of_points, np.nan), shape=number_of_points, units="deg"
        )
        self.add_input(
            "alpha", val=np.full(number_of_points, np.nan), shape=number_of_points, units="deg"
        )
        self.add_input(
            "delta_m", val=np.full(number_of_points, np.nan), shape=number_of_points, units="deg"
        )
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:cruise:CL0", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:elevator:low_speed:CL_delta", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:elevator:low_speed:CD_delta", val=np.nan, units="rad**-2")
        self.add_input("delta_Cl", val=np.full(number_of_points, np.nan))
        self.add_input("delta_Cd", val=np.full(number_of_points, np.nan))
        self.add_input("delta_Cm", val=np.full(number_of_points, np.nan))
        self.add_input(
            "thrust", val=np.full(number_of_points, np.nan), shape=number_of_points, units="N"
        )
        self.add_input(
            "thrust_rate_t", val=np.full(number_of_points, np.nan), shape=number_of_points
        )
        self.add_input("engine_setting", val=np.full(number_of_points, np.nan))
        self.add_input(
            "fuel_consumed_t",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="kg",
        )
        self.add_input(
            "non_consumable_energy_t",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="W*h",
        )
        self.add_input(
            "time_step", shape=number_of_points, val=np.full(number_of_points, np.nan), units="s"
        )

        self.add_output(
            "tsfc", shape=number_of_points, val=np.full(number_of_points, 7e-6), units="kg/s/N"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        time = inputs["time"]
        altitude = inputs["altitude"]
        distance = inputs["position"]
        mass = inputs["mass"]
        x_cg = inputs["x_cg"]
        v_tas = inputs["true_airspeed"]
        v_eas = inputs["equivalent_airspeed"]
        d_vx_dt = inputs["d_vx_dt"]
        atm = Atmosphere(altitude, altitude_in_feet=False)
        atm.true_airspeed = v_tas
        gamma = inputs["gamma"]
        alpha = inputs["alpha"] * np.pi / 180.0

        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        cl0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        coeff_k_wing = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        cl0_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL0"]
        cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
        coeff_k_htp = inputs["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]
        delta_cl = inputs["delta_Cl"]
        delta_cd = inputs["delta_Cd"]
        delta_cm = inputs["delta_Cd"]
        delta_m = inputs["delta_m"] * np.pi / 180.0
        cl_delta_m = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]
        cd_delta_m = inputs["data:aerodynamics:elevator:low_speed:CD_delta"]
        cl_wing = cl0_wing + cl_alpha_wing * alpha + delta_cl
        cl_htp = cl0_htp + cl_alpha_htp * alpha + cl_delta_m * delta_m
        cl_aircraft = cl_wing + cl_htp

        cd_tot = (
            cd0
            + delta_cd
            + coeff_k_wing * cl_wing ** 2.0
            + coeff_k_htp * cl_htp ** 2.0
            + (cd_delta_m * delta_m ** 2.0)
        )

        thrust = inputs["thrust"]
        thrust_rate = inputs["thrust_rate_t"]
        engine_setting = inputs["engine_setting"]
        fuel_consumed_t = inputs["fuel_consumed_t"]
        non_consumable_energy_t = inputs["non_consumable_energy_t"]
        time_step = inputs["time_step"]

        tsfc = fuel_consumed_t / time_step / thrust
        fuel_flow = fuel_consumed_t / time_step

        name = np.concatenate(
            (
                np.full(POINTS_NB_CLIMB, "sizing:main_route:climb"),
                np.full(POINTS_NB_CRUISE, "sizing:main_route:cruise"),
                np.full(POINTS_NB_DESCENT, "sizing:main_route:descent"),
            )
        )

        if self.options["out_file"] == "":

            outputs["tsfc"] = tsfc

        else:

            if os.path.exists(self.options["out_file"]):
                os.remove(self.options["out_file"])

            if not os.path.exists(os.path.dirname(self.options["out_file"])):
                os.mkdir(os.path.dirname(self.options["out_file"]))

            results_df = pd.DataFrame(columns=CSV_DATA_LABELS)
            results_df["time"] = time
            results_df["altitude"] = altitude
            results_df["ground_distance"] = distance
            results_df["mass"] = mass
            results_df["x_cg"] = x_cg
            results_df["true_airspeed"] = v_tas
            results_df["equivalent_airspeed"] = v_eas
            results_df["mach"] = atm.mach
            results_df["d_vx_dt"] = d_vx_dt
            results_df["density"] = atm.density
            results_df["gamma"] = gamma
            results_df["alpha"] = alpha * 180.0 / np.pi
            results_df["delta_m"] = delta_m * 180.0 / np.pi
            results_df["cl_wing"] = cl_wing
            results_df["cl_htp"] = cl_htp
            results_df["cl_aircraft"] = cl_aircraft
            results_df["cd_aircraft"] = cd_tot
            results_df["delta_Cl"] = delta_cl
            results_df["delta_Cd"] = delta_cd
            results_df["delta_Cm"] = delta_cm
            results_df["thrust (N)"] = thrust
            results_df["thrust_rate"] = thrust_rate
            results_df["engine_setting"] = engine_setting
            results_df["tsfc (kg/s/N)"] = tsfc
            results_df["energy_consumed (W*h)"] = non_consumable_energy_t
            results_df["name"] = name
            results_df["fuel_flow (kg/s)"] = fuel_flow
            results_df["time step (s)"] = time_step

            results_df.to_csv(self.options["out_file"])

            outputs["tsfc"] = tsfc

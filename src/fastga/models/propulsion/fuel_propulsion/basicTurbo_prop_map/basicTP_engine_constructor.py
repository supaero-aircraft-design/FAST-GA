"""Parametric turboprop engine map constructor."""
# -*- coding: utf-8 -*-
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
import openmdao.api as om

import fastoad.api as oad
from fastoad.constants import EngineSetting
from fastoad.module_management.constants import ModelDomain
from stdatm import Atmosphere

from fastga.models.propulsion.fuel_propulsion.basicTurbo_prop.basicTP_engine import BasicTPEngine
from fastga.models.aerodynamics.external.propeller_code.compute_propeller_aero import (
    THRUST_PTS_NB,
    SPEED_PTS_NB,
)

# Logger for this module
_LOGGER = logging.getLogger(__name__)

INVALID_SFC = -1e-2
THRUST_PTS_NB_TURBOPROP = 50
MACH_PTS_NB_TURBOPROP = 10

# Set of dictionary keys that are mapped to instance attributes.
ENGINE_LABELS = {
    "power_SL": dict(doc="power at sea level in watts."),
    "mass": dict(doc="Mass in kilograms."),
    "length": dict(doc="Length in meters."),
    "height": dict(doc="Height in meters."),
    "width": dict(doc="Width in meters."),
}
# Set of dictionary keys that are mapped to instance attributes.
NACELLE_LABELS = {
    "wet_area": dict(doc="Wet area in meters²."),
    "length": dict(doc="Length in meters."),
    "height": dict(doc="Height in meters."),
    "width": dict(doc="Width in meters."),
}


@oad.RegisterOpenMDAOSystem(
    "fastga.propulsion.turboprop_construction", domain=ModelDomain.AERODYNAMICS
)
class ComputeTurbopropMap(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "number_of_thrust_subdivision",
            default=10,
            types=int,
        )
        self.options.declare(
            "intermediate_altitude",
            default=None,
            types=float,
        )

    def setup(self):
        self.add_input("data:propulsion:turboprop:design_point:power", np.nan, units="kW")
        self.add_input(
            "data:propulsion:turboprop:design_point:turbine_entry_temperature", np.nan, units="K"
        )
        self.add_input("data:propulsion:turboprop:design_point:OPR", np.nan)
        self.add_input("data:propulsion:turboprop:design_point:altitude", np.nan, units="m")
        self.add_input("data:propulsion:turboprop:design_point:mach", np.nan)
        self.add_input("data:propulsion:turboprop:off_design:bleed_usage", np.nan)
        self.add_input("data:propulsion:turboprop:off_design:itt_limit", np.nan, units="K")
        self.add_input("data:propulsion:turboprop:off_design:power_limit", np.nan, units="kW")
        self.add_input("data:propulsion:turboprop:off_design:opr_limit", np.nan)
        self.add_input("data:TLAR:v_cruise", np.nan, units="m/s")
        self.add_input("data:aerodynamics:propeller:cruise_level:altitude", np.nan, units="m")
        self.add_input("data:geometry:propulsion:engine:layout", np.nan)
        self.add_input(
            "data:aerodynamics:propeller:sea_level:speed",
            np.full(SPEED_PTS_NB, np.nan),
            units="m/s",
        )
        self.add_input(
            "data:aerodynamics:propeller:sea_level:thrust",
            np.full(THRUST_PTS_NB, np.nan),
            units="N",
        )
        self.add_input(
            "data:aerodynamics:propeller:sea_level:thrust_limit",
            np.full(SPEED_PTS_NB, np.nan),
            units="N",
        )
        self.add_input(
            "data:aerodynamics:propeller:sea_level:efficiency",
            np.full((SPEED_PTS_NB, THRUST_PTS_NB), np.nan),
        )
        self.add_input(
            "data:aerodynamics:propeller:cruise_level:speed",
            np.full(SPEED_PTS_NB, np.nan),
            units="m/s",
        )
        self.add_input(
            "data:aerodynamics:propeller:cruise_level:thrust",
            np.full(THRUST_PTS_NB, np.nan),
            units="N",
        )
        self.add_input(
            "data:aerodynamics:propeller:cruise_level:thrust_limit",
            np.full(SPEED_PTS_NB, np.nan),
            units="N",
        )
        self.add_input(
            "data:aerodynamics:propeller:cruise_level:efficiency",
            np.full((SPEED_PTS_NB, THRUST_PTS_NB), np.nan),
        )
        self.add_input(
            "data:aerodynamics:propeller:installation_effect:effective_efficiency:low_speed",
            val=1.0,
        )
        self.add_input(
            "data:aerodynamics:propeller:installation_effect:effective_efficiency:cruise",
            val=1.0,
        )
        self.add_input(
            "data:aerodynamics:propeller:installation_effect:effective_advance_ratio",
            val=1.0,
        )
        self.add_input("settings:propulsion:turboprop:efficiency:first_compressor_stage", val=0.85)
        self.add_input("settings:propulsion:turboprop:efficiency:second_compressor_stage", val=0.86)
        self.add_input("settings:propulsion:turboprop:efficiency:high_pressure_turbine", val=0.86)
        self.add_input("settings:propulsion:turboprop:efficiency:power_turbine", val=0.86)
        self.add_input(
            "settings:propulsion:turboprop:efficiency:combustion", val=43.260e6 * 0.95, units="J/kg"
        )
        self.add_input("settings:propulsion:turboprop:efficiency:high_pressure_axe", val=0.98)
        self.add_input("settings:propulsion:turboprop:pressure_loss:inlet", val=0.8)
        self.add_input("settings:propulsion:turboprop:pressure_loss:combustion_chamber", val=0.95)
        self.add_input("settings:propulsion:turboprop:bleed:turbine_cooling", val=0.05)
        self.add_input(
            "settings:propulsion:turboprop:electric_power_offtake", val=50 * 745.7, units="W"
        )
        self.add_input("settings:propulsion:turboprop:efficiency:gearbox", val=0.98)
        self.add_input("settings:propulsion:turboprop:bleed:inter_compressor", val=0.04)
        self.add_input("settings:propulsion:turboprop:design_point:mach_exhaust", val=0.4)
        self.add_input(
            "settings:propulsion:turboprop:design_point:first_stage_pressure_ratio", val=0.25
        )

        self.add_output(
            "data:propulsion:turboprop:sea_level:mach",
            shape=MACH_PTS_NB_TURBOPROP,
        )
        self.add_output(
            "data:propulsion:turboprop:sea_level:thrust", shape=THRUST_PTS_NB_TURBOPROP, units="N"
        )
        self.add_output(
            "data:propulsion:turboprop:sea_level:thrust_limit",
            shape=MACH_PTS_NB_TURBOPROP,
            units="N",
        )
        self.add_output(
            "data:propulsion:turboprop:sea_level:sfc",
            shape=(MACH_PTS_NB_TURBOPROP, THRUST_PTS_NB_TURBOPROP),
            units="kg/s/N",
        )

        self.add_output(
            "data:propulsion:turboprop:cruise_level:mach",
            shape=MACH_PTS_NB_TURBOPROP,
        )
        self.add_output(
            "data:propulsion:turboprop:cruise_level:thrust",
            shape=THRUST_PTS_NB_TURBOPROP,
            units="N",
        )
        self.add_output(
            "data:propulsion:turboprop:cruise_level:thrust_limit",
            shape=MACH_PTS_NB_TURBOPROP,
            units="N",
        )
        self.add_output(
            "data:propulsion:turboprop:cruise_level:sfc",
            shape=(MACH_PTS_NB_TURBOPROP, THRUST_PTS_NB_TURBOPROP),
            units="kg/s/N",
        )

        self.add_output("data:propulsion:turboprop:intermediate_level:altitude", units="m")
        self.add_output(
            "data:propulsion:turboprop:intermediate_level:mach",
            shape=MACH_PTS_NB_TURBOPROP,
        )
        self.add_output(
            "data:propulsion:turboprop:intermediate_level:thrust",
            shape=THRUST_PTS_NB_TURBOPROP,
            units="N",
        )
        self.add_output(
            "data:propulsion:turboprop:intermediate_level:thrust_limit",
            shape=MACH_PTS_NB_TURBOPROP,
            units="N",
        )
        self.add_output(
            "data:propulsion:turboprop:intermediate_level:sfc",
            shape=(MACH_PTS_NB_TURBOPROP, THRUST_PTS_NB_TURBOPROP),
            units="kg/s/N",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        _LOGGER.debug("Entering turboprop computation")
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
        engine = BasicTPEngine(**engine_params)

        cruise_altitude = inputs["data:aerodynamics:propeller:cruise_level:altitude"]

        (
            mach_array_sl,
            thrust_preliminary_intersect_sl,
            sfc_general_sl,
            max_thrust_array_sl,
        ) = self.construct_table(0.0, inputs, engine)

        sfc_general_sl, thrust_preliminary_intersect_sl = format_table(
            sfc_general_sl, thrust_preliminary_intersect_sl
        )

        outputs["data:propulsion:turboprop:sea_level:mach"] = mach_array_sl
        outputs["data:propulsion:turboprop:sea_level:thrust"] = thrust_preliminary_intersect_sl
        outputs["data:propulsion:turboprop:sea_level:thrust_limit"] = max_thrust_array_sl
        outputs["data:propulsion:turboprop:sea_level:sfc"] = sfc_general_sl

        (
            mach_array_cl,
            thrust_preliminary_intersect_cl,
            sfc_general_cl,
            max_thrust_array_cl,
        ) = self.construct_table(cruise_altitude, inputs, engine)

        sfc_general_cl, thrust_preliminary_intersect_cl = format_table(
            sfc_general_cl, thrust_preliminary_intersect_cl
        )

        outputs["data:propulsion:turboprop:cruise_level:mach"] = mach_array_cl
        outputs["data:propulsion:turboprop:cruise_level:thrust"] = thrust_preliminary_intersect_cl
        outputs["data:propulsion:turboprop:cruise_level:thrust_limit"] = max_thrust_array_cl
        outputs["data:propulsion:turboprop:cruise_level:sfc"] = sfc_general_cl

        if self.options["intermediate_altitude"] is None:
            intermediate_altitude = cruise_altitude / 2.0
        else:
            intermediate_altitude = self.options["intermediate_altitude"]

        (
            mach_array_il,
            thrust_preliminary_intersect_il,
            sfc_general_il,
            max_thrust_array_il,
        ) = self.construct_table(intermediate_altitude, inputs, engine)

        sfc_general_il, thrust_preliminary_intersect_il = format_table(
            sfc_general_il, thrust_preliminary_intersect_il
        )

        outputs["data:propulsion:turboprop:intermediate_level:altitude"] = intermediate_altitude
        outputs["data:propulsion:turboprop:intermediate_level:mach"] = mach_array_il
        outputs[
            "data:propulsion:turboprop:intermediate_level:thrust"
        ] = thrust_preliminary_intersect_il
        outputs["data:propulsion:turboprop:intermediate_level:thrust_limit"] = max_thrust_array_il
        outputs["data:propulsion:turboprop:intermediate_level:sfc"] = sfc_general_il

        _LOGGER.debug("Finishing turboprop computation")

    def construct_table(self, altitude, inputs, engine):
        """Construct the sfc table for the given engine at the given altitude."""
        nb_of_mach = MACH_PTS_NB_TURBOPROP
        nb_of_thrust = self.options["number_of_thrust_subdivision"]

        cruise_velocity = inputs["data:TLAR:v_cruise"]
        cruise_altitude = inputs["data:aerodynamics:propeller:cruise_level:altitude"]

        atm_cruise = Atmosphere(cruise_altitude, altitude_in_feet=False)

        cruise_mach = cruise_velocity / atm_cruise.speed_of_sound

        # Since the cruise speed gives by construction the highest Mach number we know we will
        # never cross it, hence the following bounds for the mach array
        mach_array = np.linspace(1e-5, 1.3 * cruise_mach, nb_of_mach)
        # print("\n", mach_array)

        # We then compute the maximum thrust for those mach they are gonna be used to define the
        # thrust for which we interpolate the fuel consumption
        max_thrust_list = []

        for mach in mach_array:
            atm = Atmosphere(altitude=altitude, altitude_in_feet=False)
            atm.mach = mach
            max_thrust = engine.max_thrust(atm)
            max_thrust_list.append(float(max_thrust))

        max_thrust_array = np.array(max_thrust_list)
        # print("\n", max_thrust_array)

        # thrust_preliminary_intersect will contain the thrust at which we will interpolate our
        # data. To minimize computation time we will try to build it at relevant point while
        # keeping the overall number of points low. To do so we will create a linspace containing
        # nb_of_thrust points for each max thrust and delete all the points that overlap with
        # linspace covering lower thrust. We initialize slightly higher than the first interval
        # to ensure that this point will be kept in the first overlap
        thrust_preliminary_intersect = np.array([min(max_thrust_array) / (nb_of_thrust - 1e-5)])

        for mach in np.flip(mach_array):
            max_thrust_current_mach = max_thrust_array[np.where(mach_array == mach)[0][0]]
            # The first element of the linspace
            first_thrust_array = max_thrust_current_mach / nb_of_thrust

            current_even_spacing = np.linspace(
                first_thrust_array, max_thrust_current_mach, nb_of_thrust
            )

            # We keep values that don't overlap and add a value slightly above the maximum thrust
            # because we will need to interpolate the data to complete the table and we want a
            # point that is not too far to reduce the error because of the over-fitting
            retained_thrust_idx = np.where(
                current_even_spacing > np.amax(thrust_preliminary_intersect)
            )[0]
            retained_thrust = np.append(
                current_even_spacing[retained_thrust_idx], np.array([1.1 * max_thrust_current_mach])
            )

            thrust_preliminary_intersect = np.union1d(thrust_preliminary_intersect, retained_thrust)

        thrust_preliminary_intersect = np.union1d(thrust_preliminary_intersect, max_thrust_array)
        # print("\n", thrust_preliminary_intersect)

        # We now compute the sfc everywhere in the validity domain (thrust < thrust_max(mach))
        sfc_general = np.zeros((np.size(mach_array), np.size(thrust_preliminary_intersect)))

        for mach in mach_array:
            atm = Atmosphere(altitude, altitude_in_feet=False)
            atm.mach = mach
            for thrust in thrust_preliminary_intersect:
                thrust = np.array([thrust])
                flight_points = oad.FlightPoint(
                    mach=mach,
                    altitude=altitude,
                    engine_setting=EngineSetting.CRUISE,
                    thrust_is_regulated=True,
                    thrust_rate=0.0,
                    thrust=thrust,
                )
                if thrust > max_thrust_array[np.where(mach_array == mach)[0][0]]:
                    sfc = INVALID_SFC
                else:
                    engine.compute_flight_points(flight_points)
                    sfc = flight_points.sfc
                sfc_general[
                    np.where(mach_array == mach)[0][0],
                    np.where(thrust_preliminary_intersect == float(thrust))[0][0],
                ] = sfc

        # sfc_general_before = np.copy(sfc_general)
        # print("\n", sfc_general_before)

        valid_idx_previous_mach = np.array([])
        thrust_to_interpolate = np.zeros((1, 1))

        # We now interpolate on the data that are missing but just enough to ensure that we will
        # be able to do a 2D interpolation with value that are not INVALID_IDX
        for mach in mach_array:
            corresponding_sfc_array = sfc_general[np.where(mach_array == mach)[0]][0]

            valid_idx = np.where(corresponding_sfc_array != INVALID_SFC)[0]

            valid_thrust = thrust_preliminary_intersect[valid_idx]
            valid_sfc = corresponding_sfc_array[valid_idx]
            valid_fuel_flow = np.multiply(valid_thrust, valid_sfc)

            valid_idx_set = set(valid_idx.tolist())
            valid_idx_previous_mach_set = set(valid_idx_previous_mach.tolist())
            idx_to_interpolate = np.array(list(valid_idx_previous_mach_set - valid_idx_set))
            # print("\n", idx_to_interpolate)

            for idx in idx_to_interpolate:
                thrust_to_interpolate[0, 0] = thrust_preliminary_intersect[idx]
                predicted_fuel_flow = valid_fuel_flow[-1]
                predicted_sfc = np.divide(predicted_fuel_flow, thrust_to_interpolate)

                sfc_general[np.where(mach_array == mach)[0][0], idx] = predicted_sfc

            valid_idx_previous_mach = valid_idx

        # thrust_plot, mach_plot = np.meshgrid(thrust_preliminary_intersect, mach_array)
        # fig3d = plt.figure()
        # ax = Axes3D(fig3d)
        # ax.scatter(thrust_plot, mach_plot, sfc_general, cmap="viridis",
        #   linewidth=0.25, label="predicted behaviour")
        # ax.scatter(thrust_plot, mach_plot, sfc_general_before, cmap="viridis",
        #   linewidth=0.25, label="reference data")
        # ax.legend()
        # plt.show()

        return mach_array, thrust_preliminary_intersect, sfc_general, max_thrust_array


def format_table(sfc_table, thrust_table):
    """Reformat the sfc table to fit the OpenMDAO formalism."""
    nb_of_mach = MACH_PTS_NB_TURBOPROP
    nb_of_thrust = np.size(thrust_table)

    formatted_sfc_table = np.zeros((nb_of_mach, THRUST_PTS_NB_TURBOPROP))
    formatted_thrust_table = np.zeros(THRUST_PTS_NB_TURBOPROP)

    formatted_sfc_table[:, 0:nb_of_thrust] = sfc_table
    formatted_thrust_table[0:nb_of_thrust] = thrust_table

    # print("\n", formatted_sfc_table)
    # print("\n", formatted_thrust_table)

    return formatted_sfc_table, formatted_thrust_table

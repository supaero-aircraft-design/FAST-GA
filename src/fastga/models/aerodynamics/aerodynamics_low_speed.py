"""FAST - Copyright (c) 2016 ONERA ISAE."""
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

from openmdao.core.group import Group

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

from fastga.models.aerodynamics.external.openvsp import ComputeAEROopenvsp

# noinspection PyProtectedMember
from fastga.models.aerodynamics.external.openvsp.compute_aero_slipstream import (
    ComputeSlipstreamOpenvspSubGroup,
)
from fastga.models.aerodynamics.external.vlm import ComputeAEROvlm
from .constants import (
    SUBMODEL_CD0,
    SUBMODEL_AIRFOIL_LIFT_SLOPE,
    SUBMODEL_DELTA_HIGH_LIFT,
    SUBMODEL_DELTA_ELEVATOR,
    SUBMODEL_CL_EXTREME_CLEAN_HT,
    SUBMODEL_CL_EXTREME_CLEAN_WING,
    SUBMODEL_CL_EXTREME,
    SUBMODEL_CL_ALPHA_VT,
    SUBMODEL_CY_RUDDER,
    SUBMODEL_EFFECTIVE_EFFICIENCY_PROPELLER,
)


@oad.RegisterOpenMDAOSystem("fastga.aerodynamics.lowspeed.legacy", domain=ModelDomain.AERODYNAMICS)
class AerodynamicsLowSpeed(Group):
    """Models for low speed aerodynamics."""

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)
        self.options.declare("use_openvsp", default=False, types=bool)
        self.options.declare("compute_slipstream", default=False, types=bool)
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare("openvsp_exe_path", default="", types=str, allow_none=True)
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare("wing_airfoil", default="naca23012.af", types=str, allow_none=True)
        self.options.declare("htp_airfoil", default="naca0012.af", types=str, allow_none=True)
        self.options.declare("vtp_airfoil", default="naca0012.af", types=str, allow_none=True)

    # noinspection PyTypeChecker
    def setup(self):
        if not self.options["use_openvsp"]:
            self.add_subsystem(
                "aero_vlm",
                ComputeAEROvlm(
                    low_speed_aero=True,
                    compute_mach_interpolation=False,
                    result_folder_path=self.options["result_folder_path"],
                    airfoil_folder_path=self.options["airfoil_folder_path"],
                    wing_airfoil_file=self.options["wing_airfoil"],
                    htp_airfoil_file=self.options["htp_airfoil"],
                ),
                promotes=["*"],
            )
        else:
            self.add_subsystem(
                "aero_openvsp",
                ComputeAEROopenvsp(
                    low_speed_aero=True,
                    compute_mach_interpolation=False,
                    result_folder_path=self.options["result_folder_path"],
                    openvsp_exe_path=self.options["openvsp_exe_path"],
                    airfoil_folder_path=self.options["airfoil_folder_path"],
                    wing_airfoil_file=self.options["wing_airfoil"],
                    htp_airfoil_file=self.options["htp_airfoil"],
                ),
                promotes=["*"],
            )

        options_cd0 = {
            "airfoil_folder_path": self.options["airfoil_folder_path"],
            "low_speed_aero": True,
            "wing_airfoil_file": self.options["wing_airfoil"],
            "htp_airfoil_file": self.options["htp_airfoil"],
            "propulsion_id": self.options["propulsion_id"],
        }
        self.add_subsystem(
            "Cd0_all",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CD0, options=options_cd0),
            promotes=["*"],
        )

        option_low_speed = {"low_speed_aero": True}
        self.add_subsystem(
            "Effective_efficiency_propeller",
            oad.RegisterSubmodel.get_submodel(
                SUBMODEL_EFFECTIVE_EFFICIENCY_PROPELLER, options=option_low_speed
            ),
            promotes=["*"],
        )

        options_airfoil = {
            "airfoil_folder_path": self.options["airfoil_folder_path"],
            "wing_airfoil_file": self.options["wing_airfoil"],
            "htp_airfoil_file": self.options["htp_airfoil"],
            "vtp_airfoil_file": self.options["vtp_airfoil"],
        }
        self.add_subsystem(
            "airfoil_lift_slope",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_AIRFOIL_LIFT_SLOPE, options=options_airfoil),
            promotes=["*"],
        )

        self.add_subsystem(
            "elevator", oad.RegisterSubmodel.get_submodel(SUBMODEL_DELTA_ELEVATOR), promotes=["*"]
        )

        self.add_subsystem(
            "high_lift", oad.RegisterSubmodel.get_submodel(SUBMODEL_DELTA_HIGH_LIFT), promotes=["*"]
        )

        option_wing_airfoil = {"wing_airfoil_file": self.options["wing_airfoil"]}
        self.add_subsystem(
            "wing_extreme_cl_clean",
            oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CL_EXTREME_CLEAN_WING, options=option_wing_airfoil
            ),
            promotes=["*"],
        )

        option_htp_airfoil = {"htp_airfoil_file": self.options["htp_airfoil"]}
        self.add_subsystem(
            "htp_extreme_cl_clean",
            oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CL_EXTREME_CLEAN_HT, options=option_htp_airfoil
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "aircraft_extreme_cl",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CL_EXTREME),
            promotes=["*"],
        )

        self.add_subsystem(
            "clAlpha_vt",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CL_ALPHA_VT, options=option_low_speed),
            promotes=["*"],
        )

        self.add_subsystem(
            "Cy_Delta_rudder",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CY_RUDDER, options=option_low_speed),
            promotes=["*"],
        )

        if self.options["compute_slipstream"]:
            self.add_subsystem(
                "aero_slipstream_openvsp_ls",
                ComputeSlipstreamOpenvspSubGroup(
                    propulsion_id=self.options["propulsion_id"],
                    result_folder_path=self.options["result_folder_path"],
                    airfoil_folder_path=self.options["airfoil_folder_path"],
                    openvsp_exe_path=self.options["openvsp_exe_path"],
                    wing_airfoil_file=self.options["wing_airfoil"],
                    low_speed_aero=True,
                ),
                promotes=["data:*"],
            )

"""
    FAST - Copyright (c) 2016 ONERA ISAE
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

from openmdao.core.group import Group

from .components.cd0 import Cd0
from .components.compute_L_D_max import ComputeLDMax
from .components.compute_cnbeta_fuselage import ComputeCnBetaFuselage
from .components.clalpha_vt import ComputeClalphaVT
from .components.hinge_moments_elevator import Compute2DHingeMomentsTail, Compute3DHingeMomentsTail
from .components import ComputeMachInterpolation

from .external.vlm import ComputeAEROvlm
from .external.openvsp import ComputeAEROopenvsp
# noinspection PyProtectedMember
from .external.openvsp.compute_aero_slipstream import _ComputeSlipstreamOpenvsp

from fastoad.module_management.service_registry import RegisterOpenMDAOSystem
from fastoad.module_management.constants import ModelDomain


@RegisterOpenMDAOSystem("fastga.aerodynamics.highspeed.legacy", domain=ModelDomain.AERODYNAMICS)
class AerodynamicsHighSpeed(Group):
    """
    Models for high speed aerodynamics
    """

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)
        self.options.declare("use_openvsp", default=False, types=bool)
        self.options.declare("compute_mach_interpolation", default=False, types=bool)
        self.options.declare("compute_slipstream", default=False, types=bool)
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare('wing_airfoil', default="naca23012.af", types=str, allow_none=True)
        self.options.declare('htp_airfoil', default="naca0012.af", types=str, allow_none=True)

    # noinspection PyTypeChecker
    def setup(self):
        if not (self.options["use_openvsp"]):
            if self.options["compute_mach_interpolation"]:
                self.add_subsystem("aero_vlm",
                                   ComputeAEROvlm(
                                       low_speed_aero=False,
                                       result_folder_path=self.options["result_folder_path"],
                                       compute_mach_interpolation=True,
                                       wing_airfoil_file=self.options["wing_airfoil"],
                                       htp_airfoil_file=self.options["htp_airfoil"],
                                   ), promotes=["*"])
            else:
                self.add_subsystem("aero_vlm",
                                   ComputeAEROvlm(
                                       low_speed_aero=False,
                                       result_folder_path=self.options["result_folder_path"],
                                       compute_mach_interpolation=False,
                                       wing_airfoil_file=self.options["wing_airfoil"],
                                       htp_airfoil_file=self.options["htp_airfoil"],
                                   ), promotes=["*"])
                self.add_subsystem("mach_interpolation_roskam",
                                   ComputeMachInterpolation(
                                       wing_airfoil_file=self.options["wing_airfoil"],
                                       htp_airfoil_file=self.options["htp_airfoil"],
                                   ), promotes=["*"])
        else:
            if self.options["compute_mach_interpolation"]:
                self.add_subsystem("aero_openvsp",
                                   ComputeAEROopenvsp(
                                       low_speed_aero=False,
                                       compute_mach_interpolation=True,
                                       result_folder_path=self.options["result_folder_path"],
                                       wing_airfoil_file=self.options["wing_airfoil"],
                                       htp_airfoil_file=self.options["htp_airfoil"],
                                   ), promotes=["*"])
            else:
                self.add_subsystem("aero_openvsp",
                                   ComputeAEROopenvsp(
                                       low_speed_aero=False,
                                       compute_mach_interpolation=False,
                                       result_folder_path=self.options["result_folder_path"],
                                       wing_airfoil_file=self.options["wing_airfoil"],
                                       htp_airfoil_file=self.options["htp_airfoil"],
                                   ), promotes=["*"])
                self.add_subsystem("mach_interpolation_roskam",
                                   ComputeMachInterpolation(
                                       wing_airfoil_file=self.options["wing_airfoil"],
                                       htp_airfoil_file=self.options["htp_airfoil"],
                                   ), promotes=["*"])
        self.add_subsystem("Cd0_all",
                           Cd0(
                               low_speed_aero=False,
                               wing_airfoil_file=self.options["wing_airfoil"],
                               htp_airfoil_file=self.options["htp_airfoil"],
                               propulsion_id=self.options["propulsion_id"],
                           ), promotes=["*"])
        self.add_subsystem("L_D_max", ComputeLDMax(), promotes=["*"])
        self.add_subsystem("cnBeta_fuse", ComputeCnBetaFuselage(), promotes=["*"])
        self.add_subsystem("clAlpha_vt", ComputeClalphaVT(), promotes=["*"])
        self.add_subsystem("ch_ht_2d", Compute2DHingeMomentsTail(), promotes=["*"])
        self.add_subsystem("ch_ht_3d", Compute3DHingeMomentsTail(), promotes=["*"])
        if self.options["compute_slipstream"]:
            self.add_subsystem("aero_slipstream_openvsp",
                               _ComputeSlipstreamOpenvsp(propulsion_id=self.options["propulsion_id"],
                                                         result_folder_path=self.options["result_folder_path"],
                                                         wing_airfoil_file=self.options["wing_airfoil"],
                                                         low_speed_aero=False,
                                                         ), promotes=["*"])

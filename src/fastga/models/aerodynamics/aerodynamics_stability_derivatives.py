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

import openmdao.api as om

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

from .constants import (
    SUBMODEL_CL_Q,
    SUBMODEL_CL_ALPHA_DOT,
    SUBMODEL_CY_R,
    SUBMODEL_CY_P,
    SUBMODEL_CL_BETA,
    SUBMODEL_CL_P,
    SUBMODEL_CL_R,
    SUBMODEL_CL_AILERON,
    SUBMODEL_CL_RUDDER,
    SUBMODEL_CM_Q,
    SUBMODEL_CM_ALPHA_DOT,
    SUBMODEL_CN_AILERON,
    SUBMODEL_CN_RUDDER,
    SUBMODEL_CN_P,
    SUBMODEL_CN_R,
)


@oad.RegisterOpenMDAOSystem(
    "fastga.aerodynamics.stability_derivatives.legacy",
    domain=ModelDomain.AERODYNAMICS,
    desc="Computation of the stability derivatives",
)
class AerodynamicsStabilityDerivatives(om.Group):
    """
    Computes the aircraft aerodynamic derivatives that are not used for the sizing of the
    aircraft in cruise or low speed conditions or both depending of the user choice. It is meant
    to provide the aerodynamic derivatives necessary for the study of the stability of the
    aircraft. Can be run outside of the sizing loop for some time gain.
    """

    def initialize(self):

        self.options.declare(
            "run_low_speed_aero",
            default=True,
            types=bool,
            desc="Run the computation of the stability derivatives in low speed conditions",
        )
        self.options.declare(
            "run_cruise_aero",
            default=True,
            types=bool,
            desc="Run the computation of the stability derivatives in cruise conditions",
        )

    def setup(self):

        if self.options["run_low_speed_aero"]:
            self.add_subsystem(
                "low_speed_derivatives",
                _AerodynamicsStabilityDerivatives(low_speed_aero=True),
                promotes=["*"],
            )
        if self.options["run_cruise_aero"]:
            self.add_subsystem(
                "cruise_derivatives",
                _AerodynamicsStabilityDerivatives(low_speed_aero=False),
                promotes=["*"],
            )


class _AerodynamicsStabilityDerivatives(om.Group):
    """
    Computes the aircraft aerodynamic derivatives that are not used for the sizing of the
    aircraft either in cruise or low speed conditions. This group is used to define a FAST-OAD
    module.
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        option = {"low_speed_aero": self.options["low_speed_aero"]}

        self.add_subsystem(
            "cl_alpha_dot",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CL_ALPHA_DOT, options=option),
            promotes=["*"],
        )
        self.add_subsystem(
            "cl_q",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CL_Q, options=option),
            promotes=["*"],
        )
        self.add_subsystem(
            "cy_r",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CY_R, options=option),
            promotes=["*"],
        )
        self.add_subsystem(
            "cy_p",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CY_P, options=option),
            promotes=["*"],
        )
        self.add_subsystem(
            "cl_beta",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CL_BETA, options=option),
            promotes=["*"],
        )
        self.add_subsystem(
            "cl_p",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CL_P, options=option),
            promotes=["*"],
        )
        self.add_subsystem(
            "cl_r",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CL_R, options=option),
            promotes=["*"],
        )

        self.add_subsystem(
            "cl_delta_aileron",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CL_AILERON, options=option),
            promotes=["*"],
        )
        self.add_subsystem(
            "cl_delta_rudder",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CL_RUDDER, options=option),
            promotes=["*"],
        )
        self.add_subsystem(
            "cm_q",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CM_Q, options=option),
            promotes=["*"],
        )
        self.add_subsystem(
            "cm_alpha_dot",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CM_ALPHA_DOT, options=option),
            promotes=["*"],
        )
        self.add_subsystem(
            "cn_delta_a",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CN_AILERON, options=option),
            promotes=["*"],
        )
        self.add_subsystem(
            "cn_delta_r",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CN_RUDDER, options=option),
            promotes=["*"],
        )
        self.add_subsystem(
            "cn_p",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CN_P, options=option),
            promotes=["*"],
        )
        self.add_subsystem(
            "cn_r",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CN_R, options=option),
            promotes=["*"],
        )

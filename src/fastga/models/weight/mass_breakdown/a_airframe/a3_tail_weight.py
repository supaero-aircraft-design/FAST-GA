"""Estimation of htp weight."""
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

import fastoad.api as oad
import openmdao.api as om
from .constants import (
    SUBMODEL_HTP_MASS,
    SUBMODEL_TAIL_MASS,
    SUBMODEL_VTP_MASS,
    HTP_WEIGHT_LEGACY,
    VTP_WEIGHT_LEGACY,
    HTP_WEIGHT_GD,
    VTP_WEIGHT_GD,
    HTP_WEIGHT_TORENBEEK,
    TAIL_WEIGHT_LEGACY,
    TAIL_WEIGHT_GD,
    TAIL_WEIGHT_TORENBEEK_GD,
)


@oad.RegisterSubmodel(SUBMODEL_TAIL_MASS, TAIL_WEIGHT_LEGACY)
class ComputeTailWeight(om.Group):
    """
    Weight estimation for htp weight

    Based on a statistical analysis. See :cite:`raymer:2012` but can also be found in
    :cite:`gudmundsson:2013`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not oad.RegisterSubmodel.active_models.get(SUBMODEL_HTP_MASS):
            oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS] = HTP_WEIGHT_LEGACY

        if not oad.RegisterSubmodel.active_models.get(SUBMODEL_VTP_MASS):
            oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS] = VTP_WEIGHT_LEGACY

    def setup(self):
        self.add_subsystem(
            "empennage_htp_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HTP_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "empennage_vtp_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_VTP_MASS),
            promotes=["*"],
        )


@oad.RegisterSubmodel(SUBMODEL_TAIL_MASS, TAIL_WEIGHT_GD)
class ComputeTailWeightGD(om.Group):
    """
    Weight estimation for htp weight

    Based on a statistical analysis. See :cite:`roskampart5:1985` traditionally used on
    commercial aircraft but found to work fairly well on high performance GA aircraft.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not oad.RegisterSubmodel.active_models.get(SUBMODEL_HTP_MASS):
            oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS] = HTP_WEIGHT_GD

        if not oad.RegisterSubmodel.active_models.get(SUBMODEL_VTP_MASS):
            oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS] = VTP_WEIGHT_GD

    def setup(self):
        self.add_subsystem(
            "empennage_htp_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HTP_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "empennage_vtp_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_VTP_MASS),
            promotes=["*"],
        )


@oad.RegisterSubmodel(SUBMODEL_TAIL_MASS, TAIL_WEIGHT_TORENBEEK_GD)
class ComputeTailWeightTorenbeekGD(om.Group):
    """
    Weight estimation for htp weight

    Based on a statistical analysis. See :cite:`roskampart5:1985` traditionally used on
    commercial aircraft but found to work fairly well on high performance GA aircraft. Should
    only be used with aircraft having a diving speed above 250 KEAS.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not oad.RegisterSubmodel.active_models.get(SUBMODEL_HTP_MASS):
            oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS] = HTP_WEIGHT_TORENBEEK

        if not oad.RegisterSubmodel.active_models.get(SUBMODEL_VTP_MASS):
            oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS] = VTP_WEIGHT_GD

    def setup(self):
        self.add_subsystem(
            "empennage_htp_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HTP_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "empennage_vtp_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_VTP_MASS),
            promotes=["*"],
        )

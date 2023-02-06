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

# pylint: disable=unused-import
from .analysis_and_plots import (
    aircraft_geometry_plot,
    evolution_diagram,
    compressibility_effects_diagram,
    cl_wing_diagram,
    drag_breakdown_diagram,
    aircraft_polar,
    cg_lateral_diagram,
    mass_breakdown_bar_plot,
    mass_breakdown_sun_plot,
    payload_range,
)

# pylint: disable=unused-import
from .load_analysis.analysis_and_plots_la import (
    force_repartition_diagram,
    rbm_diagram,
    shear_diagram,
)

# pylint: disable=unused-import
from .propeller.analysis_and_plots_propeller import (
    propeller_efficiency_map_plot,
    propeller_coeff_map_plot,
)

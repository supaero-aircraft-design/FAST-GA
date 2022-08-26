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

import os.path as pth
import shutil

# noinspection PyProtectedMember
from fastoad._utils.files import make_parent_dir


def copy_resource_from_path(source: str, resource: str, target_path):
    """
    Copies the indicated resource file to provided target path.

    If target_path matches an already existing folder, resource file will be copied in this folder
    with same name. Otherwise, target_path will be the used name of copied resource file

    :param source: source of the resource to copy
    :param resource: name of the resource to copy
    :param target_path: file system path
    """
    make_parent_dir(target_path)
    shutil.copy(pth.join(source, resource), target_path)

=========
Changelog
=========

Version 0.1.4-beta
==================

- Changed the variables that define nacelle position, they are now arrays
- When reading the .csv polar Mach number within a given precision are now read
- Variable descriptions were added
- Minor model error corrections
- Like the **generate_xml_file** was added which creates a default input file that matches what is need in **generate_configuration_file**
- Added polar computation
- Added payload-range diagram computation
- Use of the mission builder feature in FAST-OAD-GA is now possible
- Changed the name of some variables to make the use of the mission builder possible namely: data:geometry:propulsion:count, data:geometry:propulsion:layout, data:geometry:propulsion:y_ratio

Version 0.1.3-beta
==================

- NACA .csv polar files replaced to correct xfoil_polar.py read issues
- Correction of security issues using **exec** and **eval** commands

Version 0.1.2-beta
==================

- Unitary tests based on converged OAD aircraft .XML
- OAD process (integration test and API) switched to VLM method to work on linux/mac os
- Minor changes in Notebooks

Version 0.1.0-beta
==================

- First beta-release


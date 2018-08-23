import os
import sys
from tests import models_for_testing as m4t

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(PACKAGE_DIR)


fd_test = m4t.initialise_foundation_test_data()
sl_test = m4t.initialise_soil_test_data()
hz_test = m4t.initialise_hazard_test_data()
fb_test = m4t.initialise_frame_building_test_data()
wb_test = m4t.initialise_wall_building_test_data()
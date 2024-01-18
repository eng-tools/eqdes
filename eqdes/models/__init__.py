from sfsimodels import output as mo
from .frame_building import FrameBuilding, DesignedRCFrame, DesignedSFSIRCFrame, AssessedRCFrame, AssessedSFSIRCFrame
from .wall_building import WallBuilding, Deprecated_DispBasedRCWall, DispBasedRCWall
from .hazard import Hazard
from .soil import Soil
from .foundation import RaftFoundation, PadFoundation


def push_building_to_table(bd, table_name="af-table"):
    para = mo.output_to_table(bd, olist="inputs")
    if hasattr(bd, 'fd'):
        para += mo.output_to_table(bd.fd, prefix="Foundation ")
        para += mo.output_to_table(bd.sl, prefix="Soil ")
    if hasattr(bd, 'hz'):
        para += mo.output_to_table(bd.hz, prefix="Hazard ")
    para = mo.add_table_ends(para, 'latex', table_name, table_name)
    return para
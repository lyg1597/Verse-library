thermo_controller_codestring = "\
from enum import Enum, auto\n\
import copy\n\
\n\
\n\
class ThermoMode(Enum):\n\
    ON = auto()\n\
    OFF = auto()\n\
\n\
\n\
class State:\n\
    temp = 0.0\n\
    total_time = 0.0\n\
    cycle_time = 0.0\n\
    thermo_mode: ThermoMode = ThermoMode.ON\n\
\n\
    def __init__(self, temp, total_time, cycle_time, thermo_mode: ThermoMode):\n\
        pass\n\
\n\
\n\
def decisionLogic(ego: State):\n\
    output = copy.deepcopy(ego)\n\
    if ego.thermo_mode == ThermoMode.ON:\n\
        if ego.cycle_time >= 1.0 and ego.cycle_time < 1.1:\n\
            output.thermo_mode = ThermoMode.OFF\n\
            output.cycle_time = 0.0\n\
    if ego.thermo_mode == ThermoMode.OFF:\n\
        if ego.cycle_time >= 1.0 and ego.cycle_time < 1.1:\n\
            output.thermo_mode = ThermoMode.ON\n\
            output.cycle_time = 0.0\n\
    return output\n\
"
# coding: utf-8
import os
import pickle

from npi.selection.config import FIELD_ROW, FIELD_WIDTH, FIELD_DEPTH
from npi.selection.lib import SelectionsortEnv, SelectionsortProgramSet, SelectionsortTeacher, create_char_map, create_questions, run_npi
from npi.selection.model import SelectionsortNPIModel
from npi.core import ResultLogger, RuntimeSystem
from npi.terminal_core import TerminalNPIRunner, Terminal


def main(filename: str, model_path: str):
    system = RuntimeSystem()
    program_set = SelectionsortProgramSet()

    with open(filename, 'rb') as f:
        steps_list = pickle.load(f)

    npi_model = SelectionsortNPIModel(system, model_path, program_set)
    npi_model.fit(steps_list)


if __name__ == '__main__':
    import sys
    DEBUG_MODE = os.environ.get('DEBUG')
    train_filename = sys.argv[1]
    model_output = sys.argv[2]
    main(train_filename, model_output)


# coding: utf-8
import os
import curses
import pickle
from copy import copy

from npi.bubble.config import FIELD_ROW, FIELD_WIDTH, FIELD_DEPTH
from npi.bubble.lib import BubblesortEnv, BubblesortProgramSet, BubblesortTeacher, create_char_map, create_questions, run_npi
from npi.core import ResultLogger
from npi.terminal_core import TerminalNPIRunner, Terminal


def main(stdscr, filename: str, num: int, result_logger: ResultLogger):
    terminal = Terminal(stdscr, create_char_map())
    terminal.init_window(FIELD_WIDTH, FIELD_ROW)
    program_set = BubblesortProgramSet()
    bubblesort_env = BubblesortEnv(FIELD_ROW, FIELD_WIDTH, FIELD_DEPTH)

    questions = create_questions(number=num)
    teacher = BubblesortTeacher(program_set)
    npi_runner = TerminalNPIRunner(terminal, teacher)
    npi_runner.verbose = DEBUG_MODE
    steps_list = []
    f = open("debug_log.csv", "w")
    for data in questions:
        bubblesort_env.reset()
        q = copy(data)
        run_npi(bubblesort_env, npi_runner, program_set.BUBBLESORT, data)
        steps_list.append({"q": q, "steps": npi_runner.step_list})
        for step in npi_runner.step_list:
            f.write("{},".format(step.input.env[0]))
            f.write("{},".format(step.input.env[1]))
            f.write("{},".format(step.input.env[2]))
            f.write("{},".format(step.input.env[3]))
            f.write("{},".format(step.input.env[4]))
            f.write("{},".format(step.input.env[5]))
            f.write("{},".format(step.input.arguments))
            f.write("{},".format(step.input.program))
            f.write("{},".format(step.output.program))
            f.write("{},".format(step.output.r))
            f.write("{},".format(step.output.arguments))
            f.write("{}\n".format(q))
        result_logger.write(data)
        terminal.add_log(data)
    f.close()

    if filename:
        with open(filename, 'wb') as f:
            pickle.dump(steps_list, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    import sys
    DEBUG_MODE = os.environ.get('DEBUG')
    if DEBUG_MODE:
        output_filename = None
        num_data = 3
        log_filename = 'result.log'
    else:
        output_filename = sys.argv[1] if len(sys.argv) > 1 else None
        num_data = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        log_filename = sys.argv[3] if len(sys.argv) > 3 else 'result.log'
    curses.wrapper(main, output_filename, num_data, ResultLogger(log_filename))
    print("create %d training data" % num_data)

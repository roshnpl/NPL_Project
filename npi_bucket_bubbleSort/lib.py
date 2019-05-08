# coding: utf-8
import random

import numpy as np

from npi.core import Program, IntegerArguments, StepOutput, NPIStep, PG_CONTINUE, PG_RETURN, NULL
from npi.terminal_core import Screen, Terminal

__author__ = 'Junling Chen & Xingye Xu'


class BubblesortEnv:
    """
    Environment of Bubblesort
    """
    def __init__(self, height, width, num_chars):
        self.screen = Screen(height, width)         
        self.num_chars = num_chars                  
        self.pointers = [NULL] * 3                 
        self.reset()                                


    def reset(self):
        self.screen.fill(NULL)                         
        self.pointers = [NULL] * 3                      

    def get_observation(self) -> np.ndarray:
        value = []
        for ptr_kind in range(len(self.pointers)):
            value.append(self.to_one_hot(self.screen[0, self.pointers[ptr_kind]])) 
        for ptr_kind in range(len(self.pointers)):
            value.append(self.to_one_hot(self.pointers[ptr_kind])) 
        return np.array(value)  

    def to_one_hot(self, ch):                       
        ret = np.zeros((self.num_chars,), dtype=np.int8)
        if 0 <= ch < self.num_chars:
            ret[ch] = 1
        else:
            raise IndexError("ch must be 0 <= ch < %s, but %s" % (self.num_chars, ch))
        return ret

    def setup_problem(self, num_array):
        for i, s in enumerate(num_array):
            self.screen[0, i] = int(s) + 1

    def move_pointer(self, ptr_kind, left_or_right):     
        if 0 <= ptr_kind < len(self.pointers):
            self.pointers[ptr_kind] += 1 if left_or_right == 1 else -1  
            self.pointers[ptr_kind] %= self.screen.width

    def get_output(self):                           
        s = []
        for ch in self.screen[0]:                   
            if ch > 0:
                s.append(int(ch-1))
        return s

    def swap_point(self):
        ch = self.screen[0, self.pointers[0]]
        self.screen[0, self.pointers[0]] = self.screen[0, self.pointers[1]]
        self.screen[0, self.pointers[1]] = ch


class MovePtrProgram(Program):      
    output_to_env = True
    PTR_POINTER1 = 0
    PTR_POINTER2 = 1
    PTR_POINTER3 = 2

    TO_LEFT = 0
    TO_RIGHT = 1

    def do(self, env: BubblesortEnv, args: IntegerArguments):
        row = args.decode_at(0)
        left_or_right = args.decode_at(1)
        env.move_pointer(row, left_or_right)


class SwapProgram(Program):         
    output_to_env = True

    def do(self, env: BubblesortEnv, args: IntegerArguments):
        env.swap_point()


class BubblesortProgramSet:
    NOP = Program('NOP')
    MOVE_PTR = MovePtrProgram('MOVE_PTR', 3, 2)     
    SWAP = SwapProgram('SWAP')  
    BUBBLESORT = Program('BUBBLESORT')
    BUBBLE = Program('BUBBLE')
    RESET = Program('RESET')
    BSTEP = Program('BSTEP')
    COMPSWAP = Program('COMPSWAP')
    LSHIFT = Program('LSHIFT')
    RSHIFT = Program('RSHIFT')


    def __init__(self):
        self.map = {}
        self.program_id = 0
        self.register(self.NOP)
        self.register(self.MOVE_PTR)
        self.register(self.SWAP)
        self.register(self.BUBBLESORT)
        self.register(self.BUBBLE)
        self.register(self.RESET)
        self.register(self.BSTEP)
        self.register(self.COMPSWAP)
        self.register(self.LSHIFT)
        self.register(self.RSHIFT)

    def register(self, pg: Program):
        pg.program_id = self.program_id
        self.map[pg.program_id] = pg
        self.program_id += 1

    def get(self, i: int):
        return self.map.get(i)


class BubblesortTeacher(NPIStep):
    def __init__(self, program_set: BubblesortProgramSet):
        self.pg_set = program_set
        self.step_queue = None
        self.step_queue_stack = []
        self.sub_program = {}
        self.register_subprogram(program_set.MOVE_PTR  , self.pg_primitive)
        self.register_subprogram(program_set.SWAP      , self.pg_primitive)
        self.register_subprogram(program_set.BUBBLESORT, self.pg_bubblesort)
        self.register_subprogram(program_set.BUBBLE    , self.pg_bubble)
        self.register_subprogram(program_set.RESET     , self.pg_reset)
        self.register_subprogram(program_set.BSTEP     , self.pg_bstep)
        self.register_subprogram(program_set.COMPSWAP  , self.pg_compswap)
        self.register_subprogram(program_set.LSHIFT    , self.pg_lshift)
        self.register_subprogram(program_set.RSHIFT    , self.pg_rshift)

    def reset(self):
        super(BubblesortTeacher, self).reset()
        self.step_queue_stack = []
        self.step_queue = None

    def register_subprogram(self, pg, method):
        self.sub_program[pg.program_id] = method

    @staticmethod
    def decode_params(env_observation: np.ndarray, arguments: IntegerArguments):    
        return env_observation.argmax(axis=1), arguments.decode_all()

    def enter_function(self):
        self.step_queue_stack.append(self.step_queue or [])
        self.step_queue = None

    def exit_function(self):
        self.step_queue = self.step_queue_stack.pop()

    def step(self, env_observation: np.ndarray, pg: Program, arguments: IntegerArguments) -> StepOutput:
        if not self.step_queue:
            self.step_queue = self.sub_program[pg.program_id](env_observation, arguments)
        if self.step_queue:
            ret = self.convert_for_step_return(self.step_queue[0])
            self.step_queue = self.step_queue[1:]
        else:
            ret = StepOutput(PG_RETURN, None, None)
        return ret

    @staticmethod
    def convert_for_step_return(step_values: tuple) -> StepOutput:
        if len(step_values) == 2:
            return StepOutput(PG_CONTINUE, step_values[0], IntegerArguments(step_values[1]))
        else:
            return StepOutput(step_values[0], step_values[1], IntegerArguments(step_values[2]))

    @staticmethod
    def pg_primitive(env_observation: np.ndarray, arguments: IntegerArguments):
        return None


    def pg_bubblesort(self, env_observation: np.ndarray, arguments: IntegerArguments):
        ret = []
        (pointer1, pointer2, pointer3, pointer1_pos, pointer2_pos, pointer3_pos), (a1, a2, a3) = self.decode_params(env_observation, arguments)
        if pointer3 == NULL:
            return None
        ret.append((self.pg_set.BUBBLE, None))
        ret.append((self.pg_set.RESET, None))
        return ret

    def pg_bubble(self, env_observation: np.array, arguments: IntegerArguments):
        ret = []
        p = self.pg_set
        (pointer1, pointer2, pointer3, pointer1_pos, pointer2_pos, pointer3_pos), (a1, a2, a3) = self.decode_params(env_observation, arguments)
        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_POINTER2, p.MOVE_PTR.TO_RIGHT)))
        ret.append((p.BSTEP, None))
        ret[-1] = (PG_RETURN, ret[-1][0], ret[-1][1])
        return ret

    def pg_reset(self, env_observation: np.array, arguments: IntegerArguments):
        ret = []
        p = self.pg_set
        ret.append((self.pg_set.LSHIFT, None))
        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_POINTER1, p.MOVE_PTR.TO_RIGHT)))
        ret.append((PG_RETURN, p.MOVE_PTR, (p.MOVE_PTR.PTR_POINTER3, p.MOVE_PTR.TO_RIGHT)))
        return ret

    def pg_bstep(self, env_observation: np.array, arguments: IntegerArguments):
        ret = []
        (pointer1, pointer2, pointer3, pointer1_pos, pointer2_pos, pointer3_pos), (a1, a2, a3) = self.decode_params(env_observation, arguments)
        if pointer2 == NULL:
            return None
        if self.compare(pointer1, pointer2) == 1:
            ret.append((self.pg_set.COMPSWAP, None))
        ret.append((self.pg_set.RSHIFT, None))
        return ret

    def pg_compswap(self, env_observaiton: np.array, arguments: IntegerArguments):
        ret = []
        p = self.pg_set
        (pointer1, pointer2, pointer3, pointer1_pos, pointer2_pos, pointer3_pos), (a1, a2, a3) = self.decode_params(env_observaiton, arguments)
        ret.append((PG_RETURN, p.SWAP, None))
        return ret

    @staticmethod
    def compare(p1, p2):
        if p1 > p2:
            return 1
        else:
            return 0

    def pg_lshift(self, env_observation: np.ndarray, arguments: IntegerArguments):
        ret = []
        p = self.pg_set
        (pointer1, pointer2, pointer3, pointer1_pos, pointer2_pos, pointer3_pos), (a1, a2, a3) = self.decode_params(env_observation, arguments)
        if pointer1 == NULL:
            return None
        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_POINTER1, p.MOVE_PTR.TO_LEFT)))
        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_POINTER2, p.MOVE_PTR.TO_LEFT)))
        return ret

    def pg_rshift(self, env_observation: np.ndarray, arguments: IntegerArguments):
        ret = []
        p = self.pg_set
        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_POINTER1, p.MOVE_PTR.TO_RIGHT)))
        ret.append((PG_RETURN, p.MOVE_PTR, (p.MOVE_PTR.PTR_POINTER2, p.MOVE_PTR.TO_RIGHT)))
        return ret


def create_char_map():
    char_map = dict((i+1, "%s" % i) for i in range(10))
    char_map[0] = ' '
    return char_map


def random_int_list(start = 0, stop = 9, length = 5):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        ri = random.randint(start, stop)
        while ri in random_list:
            ri = random.randint(start, stop)
        random_list.append(ri)
    return random_list

def create_random_questions(start = 0, stop = 0, number = 1000, maxlength = 9):
    questions = []
    for _ in range(number):
        random_length = random.randint(1, maxlength)
        question = random_int_list(start, stop, random_length)
        questions.append(dict(raw=question))
    return questions

def create_questions(start = 0, stop = 9, number = 1000, maxlength = 9):
    questions = []
    for i in range(10):
        question = [i];
        questions.append(dict(raw=question))
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            question = [i, j];
            questions.append(dict(raw=question))
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            for k in range(10):
                if i == k or j == k:
                    continue
                question = [i, j, k]
                questions.append(dict(raw=question))
    for _ in range(number):
        random_length = random.randint(4, maxlength)
        question = random_int_list(start, stop, random_length)
        questions.append(dict(raw=question))
    return questions


def run_npi(bubblesort_env, npi_runner, program, data):
    data['expect'] = sorted(data['raw'])

    bubblesort_env.setup_problem(data['raw'])

    npi_runner.reset()
    npi_runner.display_env(bubblesort_env, force=True)
    npi_runner.npi_program_interface(bubblesort_env, program, IntegerArguments())

    data['result'] = bubblesort_env.get_output()
    data['correct'] = data['result'] == data['expect']

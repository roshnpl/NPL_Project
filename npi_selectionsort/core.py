# coding: utf-8

import json
from copy import copy

import numpy as np

MAX_ARG_NUM = 3
ARG_DEPTH = 10   # 0~9 digit. one-hot.

PG_CONTINUE = 0
PG_RETURN = 1

NULL = 0


class IntegerArguments:
    depth = ARG_DEPTH
    max_arg_num = MAX_ARG_NUM
    size_of_arguments = depth * max_arg_num

    def __init__(self, args: list=None, values: np.ndarray=None):       #将value填充成一个3*10的二维数组，是3个0~9的数组合而成,
        if values is not None:                                          #最后一个是原子操作的序号，前两个是原子操作的参数，因为数量小可以采用编码
            self.values = values.reshape((self.max_arg_num, self.depth))
        else:
            self.values = np.zeros((self.max_arg_num, self.depth), dtype=np.float32)

        if args:
            for i, v in enumerate(args):                #如果args是数组形式，则i应该是数组中数据的下表，v是数组中的数据
                self.update_to(i, v)                    #args最多是一个3个数组成的表，update会依据args的值修改value的值

    def copy(self):
        obj = IntegerArguments()
        obj.values = np.copy(self.values)
        return obj

    def decode_all(self):                           #为整个value解码成int
        return [self.decode_at(i) for i in range(len(self.values))]

    def decode_at(self, index: int) -> int:         #为value中的一个值解码成int
        return self.values[index].argmax()

    def update_to(self, index: int, integer: int):
        self.values[index] = 0
        self.values[index, int(np.clip(integer, 0, self.depth-1))] = 1      #clip减除函数，超过范围的数都被置为范围边界

    def __str__(self):
        return "<IA: %s>" % self.decode_all()


class Program:
    output_to_env = False

    def __init__(self, name, *args):
        self.name = name
        self.args = args
        self.program_id = None

    def description_with_args(self, args: IntegerArguments) -> str:
        int_args = args.decode_all()
        return "%s(%s)" % (self.name, ", ".join([str(x) for x in int_args]))

    def to_one_hot(self, size, dtype=np.float):
        ret = np.zeros((size,), dtype=dtype)
        ret[self.program_id] = 1
        return ret

    def do(self, env, args: IntegerArguments):
        raise NotImplementedError()

    def __str__(self):
        return "<Program: name=%s>" % self.name


class StepInput:
    def __init__(self, env: np.ndarray, program: Program, arguments: IntegerArguments):
        self.env = env
        self.program = program
        self.arguments = arguments


class StepOutput:
    def __init__(self, r: float, program: Program=None, arguments: IntegerArguments=None):
        self.r = r
        self.program = program
        self.arguments = arguments

    def __str__(self):
        return "<StepOutput: r=%s pg=%s arg=%s>" % (self.r, self.program, self.arguments)


class StepInOut:
    def __init__(self, input: StepInput, output: StepOutput):
        self.input = input
        self.output = output


class ResultLogger:
    def __init__(self, filename):
        self.filename = filename

    def write(self, obj):
        with open(self.filename, "a") as f:
            json.dump(obj, f)
            f.write("\n")


class NPIStep:
    def reset(self):
        pass

    def enter_function(self):
        pass

    def exit_function(self):
        pass

    def step(self, env_observation: np.ndarray, pg: Program, arguments: IntegerArguments) -> StepOutput:
        raise NotImplementedError()


class RuntimeSystem:
    def __init__(self, terminal=None):
        self.terminal = terminal

    def logging(self, message):
        if self.terminal:
            self.terminal.add_log(message)
        else:
            print(message)


def to_one_hot_array(idx, size, dtype=np.int8):
    ret = np.zeros((size, ), dtype=dtype)
    ret[idx] = 1
    return ret

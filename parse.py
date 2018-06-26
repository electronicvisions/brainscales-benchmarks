#!/usr/bin/env python

import json
import subprocess
import itertools as it

benchmarks = json.load(open("benchmarks.json", "r"))

for item in benchmarks:
    name = item["model"]["name"]
    basecommand = item['tasks']['command']
    arguments = item['tasks']['arguments']
    argnames = []
    argvalues = []
    for argumentname, argumentvalues in arguments.items():
        argnames.append(argumentname)
        argvalues.append(argumentvalues)
    for argtuple in it.product(*argvalues):
        argstr = " "
        for i, arg in enumerate(argtuple):
            argstr += "{} {} ".format(argnames[i], arg)
        command = basecommand + argstr + "--name {}".format(name)
        print(command)
        # TODO: make parallel and option to use w/out slurm
        subprocess.check_call(["srun", "-p", "jenkins", "python"] + command.split(" "))

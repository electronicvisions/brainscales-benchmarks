#!/usr/bin/env python

import argparse
import itertools as it
import json
import multiprocessing as mp
import subprocess
import traceback


def run(args):
    """The args tuple needs to be formatted as follows:
    args[0]: tuple of values for the arguments
    args[1]: tuple of command line arguments
    args[2]: basecommand to call on the commandline with above arguments
    args[3]: name of the mapping problem
    args[4]: useslurm"""
    argtuple, argnames, basecommand, name, useslurm = args
    argstr = " "
    for argname, argvalue in zip(argnames, argtuple):
        argstr += "{} {} ".format(argname, argvalue)
    command = basecommand + argstr + "--name {}".format(name)
    print(command)
    if useslurm:
        try:
            subprocess.check_call(["srun", "-p", "jenkins", "python"] +
                                  command.split(" "))
        except:
            raise Exception('ERROR: {}: {}'.format(
                                                command,
                                                traceback.format_exc()))
    else:
        try:
            subprocess.check_call(["python"] + command.split(" "))
        except:
            raise Exception('ERROR: {}: {}'.format(
                                                command,
                                                traceback.format_exc()))


parser = argparse.ArgumentParser()
parser.add_argument('--useslurm', action='store_true', default=False)
parser.add_argument('--multiprocessing', action='store_true', default=False)
parser.add_argument('--processes', default=20, type=int,
                    help='Number of processes to spawn slurm jobs with, should'
                         ' be larger than number of concurrent jenkins'
                         ' slurm-jobs (at time of writing: 6). An increased'
                         ' number does not particularly hurt, as the processes'
                         ' will be blocked by the srun call anyways.')
args = parser.parse_args()

benchmarks = json.load(open("benchmarks.json", "r"))

if args.multiprocessing:
    # as the pool only holds processes waiting for srun to return
    # the number is not particularly critical, should be larger than
    # number of available slurm jobs
    pool = mp.Pool(processes=args.processes)

argtuples = []

for item in benchmarks:
    name = item["model"]["name"]
    basecommand = item['tasks']['command']
    arguments = item['tasks']['arguments']
    argnames = []
    argvalues = []
    for argumentname, argumentvalues in arguments.items():
        argnames.append(argumentname)
        argvalues.append(argumentvalues)

    if args.multiprocessing:
        argtuples += [(at, argnames, basecommand, name, args.useslurm)
                      for at in it.product(*argvalues)]
    else:
        for argtuple in it.product(*argvalues):
            run((argtuple, argnames, basecommand, name, args.useslurm))

if args.multiprocessing:
    pool.map(run, argtuples)
    pool.close()

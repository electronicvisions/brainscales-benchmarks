#!/usr/bin/env python

import argparse
from datetime import datetime
import json

import pyhmf as pynn
import pymarocco

from pysthal.command_line_util import init_logger
init_logger("WARN", [])

class RandomNetwork(object):
    def __init__(self, N, prob, marocco, model=pynn.EIF_cond_exp_isfa_ista):
        self.N = N
        self.prob = prob
        self.model = model
        self.marocco = marocco

        pynn.setup(marocco=self.marocco)

    def build(self):

        self.neurons = pynn.Population(self.N, self.model)

        connector = pynn.FixedProbabilityConnector(
                p_connect=self.prob,
                allow_self_connections=True,
                weights=0.003)

        proj = pynn.Projection(
                self.neurons,
                self.neurons,
                connector,
                target='excitatory',
                rng=pynn.NativeRNG(42))

    def run(self):
        pynn.run(1)
        pynn.end()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prob', default=0.1, type=float)
    parser.add_argument('--N',  default=5000, type=int)

    args = parser.parse_args()

    marocco = pymarocco.PyMarocco()
    marocco.continue_despite_synapse_loss = True
    marocco.calib_backend = pymarocco.PyMarocco.CalibBackend.Default

    start = datetime.now()
    r = RandomNetwork(args.N, args.prob, marocco)
    r.build()
    mid = datetime.now()
    r.run()
    end = datetime.now()

    result = {
        "model" : "random_network",
        "task" : "N{}_p{}".format(args.N, args.prob),
        "timestamp" : datetime.now().isoformat(),
        "results" : [
            {"type" : "performance",
             "name" : "setup_time",
             "value" : (end-mid).total_seconds(),
             "units" : "s",
             "measure" : "time"
         },
            {"type" : "performance",
             "name" : "total_time",
             "value" : (end-start).total_seconds(),
             "units" : "s",
             "measure" : "time"
         },
            {"type" : "performance",
             "name" : "synapses",
             "value" : marocco.stats.getSynapses()
         },
            {"type" : "performance",
             "name" : "neurons",
             "value" : marocco.stats.getNumNeurons()
         },
            {"type" : "performance",
             "name" : "synapse_loss",
             "value" : marocco.stats.getSynapseLoss()
         },
            {"type" : "performance",
             "name" : "synapse_loss_after_l1",
             "value" : marocco.stats.getSynapseLossAfterL1Routing()
         }
        ]
    }

    with open("{}_{}_results.json".format(result["model"], result["task"]), 'w') as outfile:
        json.dump(result, outfile)

if __name__ == '__main__':
    main()

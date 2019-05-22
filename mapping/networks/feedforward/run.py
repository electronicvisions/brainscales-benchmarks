#!/usr/bin/env python

import argparse
from datetime import datetime
import sys

import json

import pyhmf as pynn
import pymarocco

from pysthal.command_line_util import init_logger
init_logger("WARN", [])


class FeedforwardNetwork(object):
    def __init__(self, num_layers, conn_prob, neurons_per_layer, marocco, model=pynn.EIF_cond_exp_isfa_ista):
        self.neurons_per_layer = neurons_per_layer
        self.num_layers = num_layers
        self.conn_prob = conn_prob
        self.model = model
        self.marocco = marocco

        pynn.setup(marocco=self.marocco)

    def build(self):

        self.neurons = []
        for i in range(self.num_layers):
            self.neurons.append(pynn.Population(self.neurons_per_layer, self.model))

        connector = pynn.FixedProbabilityConnector(
            p_connect=self.conn_prob,
            allow_self_connections=False,
            weights=0.003)
        proj = []
        for i in range(1, self.num_layers):
            proj = pynn.Projection(
                self.neurons[i-1],
                self.neurons[i],
                connector,
                target='excitatory',
                rng=pynn.NativeRNG(42))

    def run(self):
        pynn.run(1.)
        pynn.end()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--conn_prob', default=1., type=float)
    parser.add_argument('--neurons_per_layer', default=200, type=int)
    parser.add_argument('--name', default="feedforward_layered_network", type=str)

    args = parser.parse_args()

    taskname = "num_layers{}_neurons_per_layer{}_conn_prob{}".format(
        args.num_layers, args.neurons_per_layer, args.conn_prob)

    marocco = pymarocco.PyMarocco()
    marocco.continue_despite_synapse_loss = True
    marocco.calib_backend = pymarocco.PyMarocco.CalibBackend.Default
    marocco.calib_path = "/wang/data/calibration/brainscales/default"
    marocco.defects_path = "/wang/data/calibration/brainscales/default"
    marocco.persist = "results_{}_{}.xml.gz".format(args.name, taskname)

    start = datetime.now()
    r = FeedforwardNetwork(args.num_layers, args.conn_prob, args.neurons_per_layer, marocco)
    r.build()
    mid = datetime.now()
    try:
        r.run()
        totsynapses = marocco.stats.getSynapses()
        totneurons = marocco.stats.getNumNeurons()
        lostsynapses = marocco.stats.getSynapseLoss()
        lostsynapsesl1 = marocco.stats.getSynapseLossAfterL1Routing()
    except RuntimeError:
        # couldn't place all populations
        totsynapses = 1
        totneurons = 1
        lostsynapses = 1
        lostsynapsesl1 = 1

    end = datetime.now()

    result = {
        "model": args.name,
        "task": taskname,
        "timestamp": datetime.now().isoformat(),
        "results": [
            {"type": "performance",
             "name": "setup_time",
             "value": (end - mid).total_seconds(),
             "units": "s",
             "measure": "time"
             },
            {"type": "performance",
             "name": "total_time",
             "value": (end - start).total_seconds(),
             "units": "s",
             "measure": "time"
             },
            {"type": "performance",
             "name": "synapses",
             "value": totsynapses
             },
            {"type": "performance",
             "name": "neurons",
             "value": totneurons
             },
            {"type": "performance",
             "name": "synapse_loss",
             "value": lostsynapses
             },
            {"type": "performance",
             "name": "synapse_loss_after_l1",
             "value": lostsynapsesl1
             },
        ],
    }

    with open("{}_{}_results.json".format(result["model"], result["task"]),
              'w') as outfile:
        json.dump(result, outfile)

    print("{}\n{}\nSynapses lost: {}; L1 synapses lost: {}; relative synapse lost: {}".format(
        sys.argv, taskname, lostsynapses, lostsynapsesl1, float(lostsynapses) / totsynapses))

if __name__ == '__main__':
    main()

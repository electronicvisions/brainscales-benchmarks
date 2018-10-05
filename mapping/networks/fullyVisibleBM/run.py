#!/usr/bin/env python

import argparse
from datetime import datetime
import json

import pyhmf as pynn
import pymarocco

from pysthal.command_line_util import init_logger
init_logger("WARN", [])


class fullyVisibleBmNetwork(object):
    def __init__(self, N, marocco, model=pynn.EIF_cond_exp_isfa_ista):
        self.N = N
        self.model = model
        self.marocco = marocco

        pynn.setup(marocco=self.marocco)

    def build(self):

        # Set the neurons
        self.neurons = pynn.Population(self.N, self.model)

        # in the fully visible BM there each neuron projects to each neuron
        # both inhibitory and excitatory to enable switching the sing of the
        # connection during eventual training
        # self connections are excluded
        # This model only sets the skeleton of the BM without the noise sources
        connector = pynn.AllToAllConnector(weights=0.003,
                                           allow_self_connections=False)
        pynn.Projection(self.neurons,
                        self.neurons,
                        connector,
                        target='excitatory')
        pynn.Projection(self.neurons,
                        self.neurons,
                        connector,
                        target='inhibitory')

    def run(self):
        pynn.run(1)
        pynn.end()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', default=5000, type=int)
    parser.add_argument('--name', default="fullyVisibleBm_network", type=str)

    args = parser.parse_args()

    taskname = "N{}".format(args.N)

    marocco = pymarocco.PyMarocco()
    marocco.continue_despite_synapse_loss = True
    marocco.calib_backend = pymarocco.PyMarocco.CalibBackend.Default
    marocco.calib_path = "/wang/data/calibration/brainscales/default"
    marocco.defects_path = "/wang/data/calibration/brainscales/default"
    marocco.persist = "results_{}_{}.xml.gz".format(args.name, taskname)

    start = datetime.now()
    r = fullyVisibleBmNetwork(args.N, marocco)
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
             }
        ]
    }

    with open("{}_{}_results.json".format(result["model"], result["task"]),
              'w') as outfile:
        json.dump(result, outfile)


if __name__ == '__main__':
    main()

#!/usr/bin/env python

import argparse
from datetime import datetime
import json

import pyhmf as pynn
import pymarocco
import pyhalco_hicann_v2 as C
from pymarocco import Defects

from pysthal.command_line_util import init_logger
init_logger("WARN", [])


class pfeilsNoiseNetwork(object):
    def __init__(self, N, K, marocco, model=pynn.EIF_cond_exp_isfa_ista):
        """
            Class to create a noise Network following Pfeil et. al. 2016.

            Keywords:
                --- N: number of neuons
                --- K: number of presynaptic partners per neuron
        """
        self.N = N
        self.K = K
        self.model = model
        self.marocco = marocco

        pynn.setup(marocco=self.marocco)

    def build(self):

        self.neurons = pynn.Population(self.N, self.model)

        connector = pynn.FixedNumberPreConnector(self.K,
                                                 weights=1,
                                                 allow_self_connections=False)

        pynn.Projection(self.neurons,
                        self.neurons,
                        connector,
                        target='excitatory',
                        rng=pynn.NativeRNG(42))

    def run(self):
        pynn.run(1)
        pynn.end()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', default=20, type=int)
    parser.add_argument('--N', default=500, type=int)
    parser.add_argument('--name', default="random_network", type=str)
    parser.add_argument('--defects_path', type=str)
    parser.add_argument('--wafer', '-w', type=int, default=33)

    args = parser.parse_args()

    taskname = "N{}_K{}_Wafer{}".format(args.N, args.K, args.wafer)

    marocco = pymarocco.PyMarocco()
    marocco.continue_despite_synapse_loss = True
    marocco.calib_backend = pymarocco.PyMarocco.CalibBackend.Default
    marocco.calib_path = "/wang/data/calibration/brainscales/default"
    marocco.default_wafer = C.Wafer(args.wafer)
    marocco.defects.backend = Defects.Backend.XML

    if args.defects_path:
        marocco.defects.path = args.defects_path
    else:
        marocco.defects.path = "/wang/data/commissioning/BSS-1/rackplace/" + str(
            args.wafer) + "/derived_plus_calib_blacklisting/current"

    marocco.persist = "results_{}_{}.xml.gz".format(args.name, taskname)

    start = datetime.now()
    r = pfeilsNoiseNetwork(args.N, args.K, marocco)
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

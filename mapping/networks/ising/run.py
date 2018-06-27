#!/usr/bin/env python

import argparse
from datetime import datetime
import json

import pyhmf as pynn
import Coordinate as C
import pymarocco

from pysthal.command_line_util import init_logger
init_logger("WARN", [])

class IsingNetwork(object):
    def __init__(self, marocco, linearsize, dimension, nbiasneurons,
                 nsources, ksources, duplicates, sourcerate,
                 model=pynn.EIF_cond_exp_isfa_ista):
        self.linearsize = linearsize
        self.dimension = dimension
        self.nbiasneurons = nbiasneurons
        self.nsources = nsources
        self.ksources = ksources
        self.duplicates = duplicates
        self.sourcerate = sourcerate
        self.model = model
        self.marocco = marocco

        pynn.setup(marocco=self.marocco)

    def build(self):
        weights  = self._create_nn_unit_weights(self.linearsize,
                self.dimension)

        self.neurons = [pynn.Population(1, self.model) 
                            for _ in range(self.linearsize ** self.dimension)]
        self.exsources = pynn.Population(self.nsources, pynn.SpikeSourcePoisson,
                                         {'rate': self.sourcerate})
        self.insources = pynn.Population(self.nsources, pynn.SpikeSourcePoisson,
                                         {'rate': self.sourcerate})
        self.biasneurons = pynn.Population(self.nbiasneurons, self.model)

        connector = pynn.FixedNumberPreConnector(
                n=self.ksources,
                weights=0.3)
        proj = pynn.Projection(
                self.exsources,
                self.neurons,
                connector,
                target='excitatory',
                rng=pynn.NativeRNG(42))
        proj = pynn.Projection(
                self.exsources,
                self.neurons,
                connector,
                target='inhibitory',
                rng=pynn.NativeRNG(42))

        connector = pynn.FixedNumberPreConnector(
                n=self.nbiasneurons,
                weights=0.4)
        proj = pynn.Projection(
                self.biasneurons,
                self.neurons,
                connector,
                target='inhibitory',
                rng=pynn.NativeRNG(42))

        for ipre, ipost, w in weights:
            connector = pynn.AllToAllConnector(weights=1)
            proj = pynn.Projection(
                self.neurons[ipre],
                self.neurons[ipost],
                connector,
                target="excitatory"
            )


    def run(self):
        pynn.run(1)
        pynn.end()

    def _create_nn_unit_weights(self, linearsize=10, dimension=2):
        # returns list of lists for weights and ndarray for bias
        weights = []
        for nid in range(linearsize**dimension):
            connlist = [(nid + o) % (linearsize**(d + 1)) +
                        int(nid / linearsize**(d + 1)) * linearsize**(d + 1)
                        for d in range(dimension)
                        for o in [linearsize**d, -linearsize**d]
                        ]
            for connid in connlist:
                weights.append((nid, connid, 1.))

        return weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--linearsize', '-l', default=5, type=int)
    parser.add_argument('--dimension', '-d', type=int, default=2)
    parser.add_argument('--nbiasneurons', '-b', type=int, default=1)
    parser.add_argument('--nsources', '-n', type=int, default=20)
    parser.add_argument('--ksources', '-k', type=int, default=5)
    parser.add_argument('--sourcerate', '-r', type=float, default=20.)
    parser.add_argument('--duplicates', '-p', type=int, default=1)
    parser.add_argument('--wafer', '-w', type=int, default=33)

    args = parser.parse_args()

    marocco = pymarocco.PyMarocco()
    marocco.continue_despite_synapse_loss = True
    marocco.calib_backend = pymarocco.PyMarocco.CalibBackend.Default
    marocco.default_wafer = C.Wafer(args.wafer)
    marocco.calib_path = "/wang/data/calibration/brainscales/default"
    marocco.defects_path = "/wang/data/calibration/brainscales/default"

    start = datetime.now()
    r = IsingNetwork(marocco,
                     linearsize=args.linearsize,
                     dimension=args.dimension,
                     nbiasneurons=args.nbiasneurons,
                     nsources=args.nsources,
                     ksources=args.ksources,
                     sourcerate=args.sourcerate,
                     duplicates=args.duplicates)
    r.build()
    mid = datetime.now()
    r.run()
    end = datetime.now()

    result = {
        "model" : "ising_network",
        "task" : "l{}_d{}_b{}_n{}_k{}_p{}_w{}".format(args.linearsize,
                                                  args.dimension,
                                                  args.nbiasneurons,
                                                  args.nsources,
                                                  args.ksources,
                                                  args.duplicates,
                                                  args.wafer),
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
    r = main()

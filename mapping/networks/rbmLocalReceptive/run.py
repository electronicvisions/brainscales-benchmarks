#!/usr/bin/env python

import argparse
from datetime import datetime
import json
import sys

import pyhmf as pynn
import pymarocco

from pysthal.command_line_util import init_logger
init_logger("WARN", [])


class rbmLocalReceptiveFieldsNetwork(object):
    def __init__(self, N, K, L, marocco, model=pynn.EIF_cond_exp_isfa_ista):
        self.N = N
        self.K = K
        self.L = L
        self.model = model
        self.marocco = marocco

        pynn.setup(marocco=self.marocco)

    def build(self):

        #####
        # Set up the neurons
        connector = pynn.AllToAllConnector(weights=0.003,
                                           allow_self_connections=False)

        # label
        labelPop = pynn.Population(self.L, self.model)

        # visible
        visiblePop = []
        for outer in range(self.N):
            neurons = []
            for inner in range(self.N):
                neurons.append(pynn.Population(1, self.model))
            visiblePop.append(neurons)

        # visible
        Nhidden = self.N - self.K + 1
        hiddenPop = []
        for outer in range(Nhidden):
            neurons = []
            for inner in range(Nhidden):
                neurons.append(pynn.Population(1, self.model))
            hiddenPop.append(neurons)
        ####

        ####
        # Set up the connections

        # between hidden and visible
        # each neuron in the hidden layer sees
        # only a local field in the visible layer
        for outerI in range(Nhidden):
            for outerJ in range(Nhidden):
                for innerI in range(self.K):
                    for innerJ in range(self.K):
                        i = outerI + innerI
                        j = outerJ + innerJ
                        pynn.Projection(visiblePop[i][j],
                                        hiddenPop[outerI][outerJ],
                                        connector,
                                        target='excitatory')
                        pynn.Projection(visiblePop[i][j],
                                        hiddenPop[outerI][outerJ],
                                        connector,
                                        target='inhibitory')
                        pynn.Projection(hiddenPop[outerI][outerJ],
                                        visiblePop[i][j],
                                        connector,
                                        target='excitatory')
                        pynn.Projection(hiddenPop[outerI][outerJ],
                                        visiblePop[i][j],
                                        connector,
                                        target='inhibitory')

        # between hidden and label
        # there is full connectivity between
        # the hidden layer and the label layer
        for outer in range(Nhidden):
            for inner in range(Nhidden):
                pynn.Projection(hiddenPop[outer][outer],
                                labelPop,
                                connector,
                                target='inhibitory')
                pynn.Projection(hiddenPop[outer][outer],
                                labelPop,
                                connector,
                                target='excitatory')
                pynn.Projection(labelPop,
                                hiddenPop[outer][outer],
                                connector,
                                target='inhibitory')
                pynn.Projection(labelPop,
                                hiddenPop[outer][outer],
                                connector,
                                target='excitatory')

    def run(self):
        pynn.run(1)
        pynn.end()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', default=10, type=int,
                        help='Edge size of the visible layer. \
                        The number of neurons in the visible\
                        layer is hence NxN')
    parser.add_argument('--K', default=8, type=int,
                        help='Edge size of the local receptive fields.\
                        K has to be larger than N.')
    parser.add_argument('--L', default=10, type=int,
                        help='Number of neurons in the label layer.')
    parser.add_argument('--name', default="fullyVisibleBm_network", type=str)

    args = parser.parse_args()

    # The edge size of the visible layer has to be larger or equal
    # than the edge size of the receptive fields
    if args.N < args.K:
        sys.exit('The edge size of the visible layer {0}'
                 ' has to be larger than the edge size {1}'
                 ' of the local receptive '
                 ' fields!'.format(args.N, args.K))

    taskname = "N{}_K{}_L{}".format(args.N,
                                    args.K,
                                    args.L)

    marocco = pymarocco.PyMarocco()
    marocco.continue_despite_synapse_loss = True
    marocco.calib_backend = pymarocco.PyMarocco.CalibBackend.Default
    marocco.calib_path = "/wang/data/calibration/brainscales/default"
    marocco.defects_path = "/wang/data/calibration/brainscales/default"
    marocco.persist = "results_{}_{}.xml.gz".format(args.name, taskname)

    start = datetime.now()
    r = rbmLocalReceptiveFieldsNetwork(args.N, args.K, args.L, marocco)
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

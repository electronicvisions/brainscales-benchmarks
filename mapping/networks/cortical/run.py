#!/usr/bin/env python

import argparse
from datetime import datetime
import json
import numpy as np

import pyhmf as pynn
import pyhalco_hicann_v2 as C
from pymarocco import PyMarocco
from pymarocco import Defects

from pysthal.command_line_util import init_logger
init_logger("ERROR", [])

import params as par

import pylogging
logger = pylogging.get("column-benchmark")

# At the moment only the deflaut placement strategy is tested. Can be added later to test different strategy
from pymarocco_runtime import ClusterByPopulationConnectivity as placer_pop
from pymarocco_runtime import ClusterByNeuronConnectivity as placer_neuron_cluster
from pymarocco_runtime import byNeuronBlockEnumAndPopulationIDasc as placer_enum_IDasc

class CorticalNetwork(object):
    def __init__(self, marocco, scale, k_scale, seed):

        # total connection counter
        self.totalConnections = 0

        self.marocco = marocco
        # scale compared to original in amount of neurons network which has about 80,000 neurons
        self.scale = scale
        # scale connections, scales number of connections
        self.k_scale = k_scale

        # Name Tag
        self.model = pynn.IF_cond_exp

        self.seed = seed

        pynn.setup(marocco=self.marocco)

    def get_indegrees(self):
        '''Get number of incoming synapses per neuron (used for in-degree scaling)'''
        K = np.zeros([len(par.label),len(par.label)])
        num_neurons = self.get_neuron_number()
        for target_index, target_pop in enumerate(par.label):
            for source_index, source_pop in enumerate(par.label):
                n_target = num_neurons[target_index]
                n_source = num_neurons[source_index]
                K[target_index][source_index] = np.log(1. -
                    par.conn_probs[target_index][source_index]) / np.log(
                    1. - 1. / (n_target * n_source))/n_target
        return K

    def get_neuron_number(self):
        '''stores the neuron numbers in list ordered such as label'''
        num_neurons = []
        layers = ['L23','L4','L5','L6']
        keys = ['E', 'I']
        for layer in layers:
            for key in keys:
                num_neurons.append(par.num_neurons[layer][key])
        return num_neurons

    def build(self):
        # set populations
        self.populations = {}

        # calculate indegrees from connection probability
        self.indegrees = self.get_indegrees()

        for layer, exIn in par.num_neurons.items():
            # [:1] to remove the first "L"
            self.populations[layer[1:] + "e"] = pynn.Population(
                int(exIn["E"] * self.scale), self.model)
            self.populations[layer[1:] + "i"] = pynn.Population(
                int(exIn["I"] * self.scale), self.model)

        # Create projections
        self.projections = []
        self.projectionLabels = []

        for targetIndex, targetPop in enumerate(par.label):
            for sourceIndex, sourcePop in enumerate(par.label):

                if sourcePop.endswith("e"):
                    target = "excitatory"
                else:
                    target = "inhibitory"

                sourceSize = self.populations[sourcePop].size
                targetSize = self.populations[targetPop].size

                # In-degree scaling as described in Albada et al. (2015) "Scalability of Asynchronous Networks
                # Is Limited by One-to-One Mapping between Effective Connectivity and Correlations"
                # Number of inputs per target neuron (in-degree) for full scale model is scaled with k_scale
                # To receive total connection number it is multiplied with downscaled target population size (scale)
                # Connection probability is not preserved if scale == k_scale (multiple connections neglected)
                n_connection = int(round(self.indegrees[targetIndex][sourceIndex] * self.k_scale * targetSize))
                self.totalConnections += n_connection
                if(n_connection == 0):
                    continue

                # connection matrix [(neuron_pop1,neuron_pop2,weight,delay),(...)]
                matrix = np.zeros((4, n_connection),dtype= float)
                np.random.seed(self.seed)
                matrix[0] = np.random.randint(0, sourceSize, n_connection)
                matrix[1] = np.random.randint(0, targetSize, n_connection)

                # The delay and weight is not important for mapping
                # PyNN requires it to be set to some value
                matrix[2] = np.repeat(1, n_connection) # arbitrary weight
                matrix[3] = np.repeat(0, n_connection) # arbitrary delay
                matrix = matrix.T
                matrix = [[int(a),int(b),c,d] for a,b,c,d in matrix]
                connector = pynn.FromListConnector(matrix)

                self.projections.append(pynn.Projection(
                    self.populations[sourcePop], self.populations[targetPop], connector, target=target, label=sourcePop + "-" + targetPop))

                self.projectionLabels.append(sourcePop + "-" + targetPop)
        print("total connections:", self.totalConnections)

        # external input:
        self.externalInputPops = {}

        # External spikes or external current
        external_source = par.external_source
        # will not work for large networks, for now it is not used due to par.external_source
        if (external_source == "spikeInput"):
            print("using external input connections")
            for layer, amount in par.K_ext.items():
                # rate is given in model with 8Hz
                # will not work for large networks, for now it is not used due to par.external_source
                rate_to_ex = par.bg_rate * amount["E"] * self.k_scale
                rate_to_in = par.bg_rate * amount["I"] * self.k_scale
                self.externalInputPops[layer[1:] + "e"] = pynn.Population(
                    self.populations[layer[1:] + "e"].size, pynn.SpikeSourcePoisson, {'rate': rate_to_ex})
                self.externalInputPops[layer[1:] + "i"] = pynn.Population(
                    self.populations[layer[1:] + "i"].size, pynn.SpikeSourcePoisson, {'rate': rate_to_in})

            # create connections
            for sourceKey, sourcePop in self.externalInputPops.items():
                # set connector for each pop size since RandomDistribution object not supported by pyhmf
                # arbitrary weight
                externalConnector = pynn.OneToOneConnector(
                    weights = 1)
                # create connection
                self.projections.append(pynn.Projection(
                    sourcePop, self.populations[sourceKey], externalConnector, target="excitatory"))
                self.projectionLabels.append("ext.-" + targetPop)

    def getLoss(self, marocco):
        perPopulation = {}
        for i in range(len(self.projections)):
            synLoss, totalSyn = self.projectionwise_synapse_loss(
                self.projections[i], marocco)
            perPopulation[self.projectionLabels[i]] = {
                "synLoss": synLoss, "TotalSyns": totalSyn}

        return perPopulation

    def run(self):
        pynn.run(1)
        pynn.end()

    def projectionwise_synapse_loss(self, proj, marocco):
        """
        computes the synapse loss of a projection
        params:
        proj    - a pyhmf.Projection
        marocco -  the PyMarocco object after the mapping has run.
        returns: (nr of lost synapses, total synapses in projection)
        """
        orig_weights = proj.getWeights(format='array')
        mapped_weights = marocco.stats.getWeights(proj)
        syns = np.where(~np.isnan(orig_weights))
        realized_syns = np.where(~np.isnan(mapped_weights))
        orig = len(syns[0])
        realized = len(realized_syns[0])
        if orig > 0:

            print ("Projection-Wise Synapse Loss", proj, (orig - realized) * 100. / orig)
        return orig - realized, orig

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser()
    # scale factor of the whole network compared to the original one
    parser.add_argument('--scale', default=0.01, type=float)
    # size of one neueron in hw neurons
    parser.add_argument('--n_size', default=4, type=int)
    parser.add_argument('--k_scale', type=float)  # scale of connections

    # wafer defects that should be considered in the mapping
    parser.add_argument('--wafer', '-w', type=int, default=24)

    # specific path where the defect parts of the wafer are saved
    # if nothing specified, current defects of the given wafer are used
    parser.add_argument('--defects_path', type=str)
    parser.add_argument('--ignore_blacklisting', type=str2bool, nargs='?',
                        default = False, const=True)
    parser.add_argument('--name', type=str,
                        default='cortical_column_network')  # name
    parser.add_argument('--placer', type=str, default='byNeuron')
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    # k_scale is set to "scale" by deflaut
    if not args.k_scale:
        args.k_scale = args.scale

    taskname = "scale{}_k-scale{}_nsize{}_wafer{}_ignoreBlacklsiting{}".format(
        args.scale,
        args.k_scale,
        args.n_size,
        args.wafer,
        args.ignore_blacklisting)

    marocco = PyMarocco()
    marocco.neuron_placement.default_neuron_size(args.n_size)

    if(args.ignore_blacklisting):
        marocco.defects.backend = Defects.Backend.Without
    else:
        marocco.defects.backend = Defects.Backend.XML


    marocco.skip_mapping = False
    marocco.backend = PyMarocco.Without

    marocco.continue_despite_synapse_loss = True
    marocco.default_wafer = C.Wafer(args.wafer)  # give wafer args
    marocco.calib_backend = PyMarocco.CalibBackend.Default
    marocco.calib_path = "/wang/data/calibration/brainscales/default"

    if args.defects_path:
        marocco.defects.path = args.defects_path
    else:
        marocco.defects.path = "/wang/data/commissioning/BSS-1/rackplace/" + str(
            args.wafer) + "/derived_plus_calib_blacklisting/current"

    # c 4189 no specification
    #taskname += "_c4189_"

    # strategy
    marocco.merger_routing.strategy(  # is now default
        marocco.merger_routing.minimize_as_possible)
    #taskname += "_minimAsPoss"

    '''
    # placement strategy
    user_strat = placer()
    taskname += "_placer"
    '''

    if args.placer == "byNeuron":
        user_strat = placer_neuron_cluster()  # cluster by neurons
        taskname += "_byNeuron"
        marocco.neuron_placement.default_placement_strategy(user_strat)

    if args.placer == "byEnum":
        user_strat = placer_enum_IDasc()  # cluster by neurons
        taskname += "_byEnum"
        marocco.neuron_placement.default_placement_strategy(user_strat)

    if args.placer == "constrained":
        # needed for 5720 with patch set 36(best results) or ps 50
        from pymarocco_runtime import ConstrainedNeuronClusterer as placer_neuron_resizer

        user_strat = placer_neuron_resizer()
        taskname += "_constrained"
        marocco.neuron_placement.default_placement_strategy(user_strat)

    # give marocco the format of the results file
    taskname += str(datetime.now())
    marocco.persist = "results_{}_{}.xml.gz".format(
        args.name, taskname)

    start = datetime.now()
    r = CorticalNetwork(marocco, scale=args.scale, k_scale=args.k_scale, seed = args.seed)
    r.build()
    mid = datetime.now()
    try:
        r.run()
        totsynapses = marocco.stats.getSynapses()
        totneurons = marocco.stats.getNumNeurons()
        lostsynapses = marocco.stats.getSynapseLoss()
        lostsynapsesl1 = marocco.stats.getSynapseLossAfterL1Routing()
        perPopulation = r.getLoss(marocco)
        print("Losses: ", lostsynapses, " of ", totsynapses, " L1Loss:",
              lostsynapsesl1, " Relative:", lostsynapses / float(totsynapses))

    except RuntimeError as err:
        # couldn't place all populations
        totsynapses = 1
        totneurons = 1
        lostsynapses = 1
        lostsynapsesl1 = 1
        logger.error(err)
    end = datetime.now()
    print("time:", end - start)
    result = {
        "model": args.name,
        "task": taskname,
        "scale": args.scale,
        "k_scale": args.k_scale,
        "n_size": args.n_size,
        "wafer": args.wafer,
        "ignore_blacklisting": args.ignore_blacklisting,
        "timestamp": datetime.now().isoformat(),
        "placer": args.placer,
        "perPopulation": perPopulation,
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
    r = main()

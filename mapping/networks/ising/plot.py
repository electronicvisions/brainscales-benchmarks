
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import json
import glob

from collections import defaultdict


plotdata = defaultdict(list)
for jsfile in glob.glob('*.json'):
    if jsfile.startswith("benchmarks"):
        continue
    name, parameters = jsfile.split("_network_")
    with open(jsfile, 'r') as f:
        data = json.load(f)
        nneurons = data['results'][3]['value']
        totsyn = data['results'][2]['value']
        losssyn = data['results'][4]['value']
        lossl1syn = data['results'][5]['value']
        plotdata[name].append([nneurons, losssyn/float(totsyn), lossl1syn/float(totsyn)])

for name in plotdata:
    pd = np.array(plotdata[name])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pd[:,0], pd[:, 1], 'x', label='synapseloss')
    ax.plot(pd[:,0], pd[:, 2], 'x', label='synapsel1loss')
    ax.legend()
    plt.savefig('{}_loss.pdf'.format(name))

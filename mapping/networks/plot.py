
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import json
import glob

from collections import defaultdict


data ={}

xkeys = ['neurons', 'synapses']
ykeys = ['synapse_loss', 'synapse_loss_after_l1']

for jsfile in glob.glob('brainscales-benchmarks/*.json'):
    if jsfile.startswith("benchmarks"):
        continue
    name, parameters = jsfile.split("_network_")
    name = name.split('/')[1]
    if name not in data:
        data[name] = defaultdict(list)
    with open(jsfile, 'r') as f:
        jsondata = json.load(f)
        for jd in jsondata['results']:
            if jd['name'] in (xkeys + ykeys):
                data[name][jd['name']].append(float(jd['value']))

plotdata = {}
for name in data.keys():
    plotdata[name] = {}
    for key in (xkeys + ykeys):
        plotdata[name][key] = np.array(data[name][key])

    for key in ykeys:
        plotdata[name][key] /= plotdata[name]['synapses']


for name in plotdata:
    pd = plotdata[name]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('#synapses')
    ax.set_ylabel('percentage of lost synapses')
    for ykey in ykeys:
        ax.plot(pd['synapses'], pd[ykey], 'x', label=ykey)
    ax.legend()
    plt.savefig('{}_loss.pdf'.format(name))

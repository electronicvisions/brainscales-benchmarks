
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import json

jsonfiles = [
				(20, "ising_network_l20_d2_b1_n20_k5_p1_w33_results.json"),
				(40, "ising_network_l40_d2_b1_n20_k5_p1_w33_results.json"),
				(80, "ising_network_l80_d2_b1_n20_k5_p1_w33_results.json"),
			]

plotdata = []
for ls, jsfile in jsonfiles:
	with open(jsfile, 'r') as f:
		data = json.load(f)
		totsyn = data['results'][2]['value']
		losssyn = data['results'][4]['value']
		lossl1syn = data['results'][5]['value']
		plotdata.append([ls, losssyn/float(totsyn), lossl1syn/float(totsyn)])

plotdata = np.array(plotdata)

plt.plot(plotdata[:,0], plotdata[:, 1], label='synapseloss')
plt.plot(plotdata[:,0], plotdata[:, 2], label='synapsel1loss')
plt.savefig('ising_loss.pdf')
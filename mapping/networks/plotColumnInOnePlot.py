import numpy as np
import json
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plotTotalLoss(experiments):
    sizes = []
    losses = []
    labels = []

    # Put experiment information in the upper 3 lists
    for exp in experiments:
        losses.append(exp["lostSynsRel"])
        sizes.append(exp["scale"])
        if(exp["ignore_blacklisting"]):
            labels.append("n_Size: " + str(exp["n_size"]) +
                       " Model: " + exp["model"] + " placer: " + exp["placer"] + " ignore blackl.")
        else:
            labels.append("n_Size: " + str(exp["n_size"]) +
                       " Model: " + exp["model"] + " placer: " + exp["placer"] + " wafer: " + str(exp["wafer"]))

    # This is only to make another axis showing the used synapses (porportional to the network_size^2)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    # For the diagramm ticks
    maxXVal = 0

    for i in list(set(labels)):

        mask = np.array(labels) == i
        xVal = np.array(sizes)[mask]
        yVal = np.array(losses)[mask]
        sorting = np.argsort(xVal)
        printLabelset = str(i).split(" ")
        printLabel = "plot total loss of N_Size: " + \
            printLabelset[1] + " Placer: " + printLabelset[5] + " " + printLabelset[6] + " " + printLabelset[7]
        print(printLabel)

        # plot percentage
        ax1.plot(100 * xVal[sorting], 100 * yVal[sorting],
                 marker=".", alpha=0.8, linewidth=0.1, label=printLabel)

        maxXVal = np.max( [ np.max(xVal), maxXVal] )

    # plot
    # get handles and labels
    current_handles, current_labels = ax1.get_legend_handles_labels()

    # sort or reorder the labels and handles
    reversed_labels, reversed_handles = zip(
        *sorted(zip(current_labels, current_handles)))

    ax1.legend(reversed_handles, reversed_labels)
    ax1.set_xlabel("Size Compared to the Original Model in %")
    ax1.set_ylabel("Relative Synapse Losses in %")

    # Calc the amount of synapses for each size
    scale = np.linspace(0, maxXVal*100, 6)
    totalSyns = 300000000
    synapseAmount = np.array((scale / 100.0)**2 *
                             totalSyns / 10000.0, dtype=int)

    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax2.set_xlabel("Amount of total Synapses in 10,000")
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(scale)
    ax1.set_xticks(scale)
    ax2.set_xticklabels(synapseAmount)

    fig.savefig("cortical_column_all_losses.pdf")
    plt.figure()

def plotLossPerConnection(experiments):
    colors = ["orange", "blue", "green", "red", "violett"]
    sortings = ["Source", "Target"]

    pdfname = 'cortical_column_population_wise_losses.pdf'
    with PdfPages(pdfname) as pdf:
        for sorting in sortings:
            for experiment in experiments:

                plt.figure(figsize=(9, 6))

                # collect all data
                loss = []
                total = []
                labels = []
                for key, value in experiment["perPopulation"].items():
                    loss.append(value["synLoss"])
                    total.append(value["TotalSyns"])
                    labels.append(key)
                loss, total, labels = np.array(
                    loss), np.array(total), np.array(labels)
                realized = total - loss
                if sorting == "Source":
                    sortMask = np.argsort(labels)

                if sorting == "Target":
                    sort = []
                    for label in labels:
                        sort.append(label.split("-")[1] + label.split("-")[0])
                    sortMask = np.argsort(np.array(sort))

                print("plotting   " + "Placer: "+ experiment["placer"] + " Scale: "+ str(100 * experiment["scale"]) +"% Neuron Size: "+ str(
                    experiment["n_size"]) )

                plt.bar(np.arange(total[sortMask].shape[0]), total[sortMask], tick_label=labels[sortMask], alpha=0.8, label="Lost Synapses")
                plt.bar(np.arange(total[sortMask].shape[0]), realized[sortMask], tick_label=labels[sortMask],  alpha=0.8, label="Realized Synapses")

                plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                plt.xticks(fontsize=8, rotation=90)
                plt.legend()
                plt.ylabel("Amount of Synapses")
                plt.title("Placer: "+ experiment["placer"] + " Scale: "+ str(100 * experiment["scale"]) +"% Neuron Size: "+ str(
                    experiment["n_size"]) + " Wafer: " + str(experiment['wafer']) +" \n Realized and Lost Synapses ordered by the " +
                        sorting + " Population", pad = 20, fontdict = {"fontsize": 14})

                plt.savefig(pdf, format='pdf')
                plt.close()

def main():

    #Load all Cortical Column Results
    files = glob.glob("cortical_column*.json")

    data = []
    for afile in files:
        with open(afile) as json_file:
            data.append(json.load(json_file))

    experiments = []
    for exper in data:

        thisRun = {}

        # read out
        thisRun["scale"] = exper["scale"]
        thisRun["model"] = exper["model"]
        thisRun["placer"] = exper["placer"]
        thisRun["n_size"] = exper["n_size"]
        thisRun["wafer"] = exper["wafer"]
        thisRun["ignore_blacklisting"] = exper["ignore_blacklisting"]
        thisRun["totSyns"] = exper["results"][2]["value"]
        thisRun["totNeurons"] = exper["results"][3]["value"]
        thisRun["lostSyns"] = exper["results"][4]["value"]
        thisRun["lostSynsl1"] = exper["results"][5]["value"]

        thisRun["lostSynsl1Rel"] = thisRun["lostSynsl1"] / \
            float(thisRun["totSyns"])
        thisRun["lostSynsRel"] = thisRun["lostSyns"] / float(thisRun["totSyns"])

        if("perPopulation" in exper):
            thisRun["perPopulation"] = exper["perPopulation"]
        experiments.append(thisRun)

    experiments = sorted(experiments, key=lambda elem: "%s %s %s %s" % (elem['placer'], elem['wafer'],elem['n_size'],elem['scale']))
    plotTotalLoss(experiments)
    plotLossPerConnection(experiments)
# Population parameters of cortical column model (cf. Potjans and Diesmann 2014)

# Population sizes
num_neurons = {
  'L23': {'E':20683, 'I':5834},
  'L4' : {'E':21915, 'I':5479},
  'L5' : {'E':4850,  'I':1065},
  'L6' : {'E':14395, 'I':2948}
}

# Probabilities for >=1 connection between neurons in the given populations. The first index is for the target population; the second for the source population
#             2/3e      2/3i    4e      4i      5e      5i      6e      6i
conn_probs = [[0.1009,  0.1689, 0.0437, 0.0818, 0.0323, 0.,     0.0076, 0.    ],
             [0.1346,   0.1371, 0.0316, 0.0515, 0.0755, 0.,     0.0042, 0.    ],
             [0.0077,   0.0059, 0.0497, 0.135,  0.0067, 0.0003, 0.0453, 0.    ],
             [0.0691,   0.0029, 0.0794, 0.1597, 0.0033, 0.,     0.1057, 0.    ],
             [0.1004,   0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.    ],
             [0.0548,   0.0269, 0.0257, 0.0022, 0.06,   0.3158, 0.0086, 0.    ],
             [0.0156,   0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
             [0.0364,   0.001,  0.0034, 0.0005, 0.0277, 0.008,  0.0658, 0.1443]]

# External connections
K_ext = {
  'L23': {'E':1600, 'I':1500},
  'L4' : {'E':2100, 'I':1900},
  'L5' : {'E':2000, 'I':1900},
  'L6' : {'E':2900, 'I':2100}
}


# names of populations
label = ["23e", "23i", "4e", "4i", "5e", "5i", "6e", "6i"]

# external DC input
DC= {'L23': {'E': 0.0, 'I': 0.0},
     'L4': {'E': 0.0, 'I': 0.0},
     'L5': {'E': 0.0, 'I': 0.0},
     'L6': {'E': 0.0, 'I': 0.0}}

# Background rate per synapse
bg_rate = 8.0

# Type of external source
external_source = "current" # "spikeInput"

# naming scheme used by Albada
layers = {'L23':0, 'L4':1, 'L5':2, 'L6':3}
pops = {'E':0, 'I':1}

structure = {'L23': {'E':0, 'I':1},
             'L4' : {'E':2, 'I':3},
             'L5' : {'E':4, 'I':5},
             'L6' : {'E':6, 'I':7}}
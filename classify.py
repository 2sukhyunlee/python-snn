import numpy as np
import os
import pandas as pd
from neuron import neuron
import random
from recep_field import rf
from spike_train import encode
from parameters import param as par
from weight_initialization import learned_weights_x, learned_weights_o, learned_weights_synapse
import imageio

#Parameters
global time, T, dt, t_back, t_fore, w_min
time  = np.arange(1, par.T+1, 1)

layer2 = []

# creating the hidden layer of neurons
for i in range(par.n):
    a = neuron()
    layer2.append(a)

#synapse matrix
synapse = np.zeros((par.n,par.m))

#learned weights
synapse[0] = learned_weights_x()
synapse[1] = learned_weights_o()

#random initialization for rest of the synapses
for i in range(par.n):
    synapse[i] = learned_weights_synapse(i)
    #for j in range(par.m):
    #    synapse[i][j] = random.uniform(0, 0.4*par.scale)

path = "mnist_png/testing/random/"
img_list = os.listdir(path)

#open file_list with excel
'''
file_df = pd.read_excel('file_list.xlsx')
file_name = file_df.loc[:, ['file_name']].tolist()
file_tag = file_df.loc[:, [file_tag]].tolist()
result_tag = []
'''

#open file_list with txt
file_name = []
file_tag = []
result_tag = []

with open('file_list.txt','r', encoding='UTF8') as f:
    lines = f.readlines()
    for line in lines:
        name, tag = line.split('\t')
        file_name.append(name)
        file_tag.append(tag[0])

total = 0
accuracy = 0

for k in range(1):

    for i, x in enumerate(img_list):
        spike_count = [0,0]

        #read the image to be classified
        img = imageio.imread(path+"/{}".format(x))

        #initialize the potentials of output neurons
        for x in layer2:
            x.initial(par.Pth)

        #calculate teh membrane potentials of input neurons
        pot = rf(img)

        #generate spike trains
        train = np.array(encode(pot))

        #flag for lateral inhibition
        f_spike = 0

        active_pot = [0,0]

        for t in time:
            for j, x in enumerate(layer2):
                active = []

        #update potential if not in refractory period
                if(x.t_rest<t):
                    x.P = x.P + np.dot(synapse[j], train[:,t])
                    if(x.P>par.Prest):
                        x.P -= par.D
                    active_pot[j] = x.P

            # Lateral Inhibition
            if(f_spike==0):
                high_pot = max(active_pot)
                if(high_pot>par.Pth):
                    f_spike = 1
                    winner = np.argmax(active_pot)
                    #print(i, winner)
                    for s in range(par.n):
                        if(s!=winner):
                            layer2[s].P = par.Prest

            #Check for spikes
            for j,x in enumerate(layer2):
                s = x.check()
                if(s==1):
                    #print(j, s)
                    spike_count[j] += 1
                    x.t_rest = t + x.t_ref

        total += 1
        result = np.argmax(active_pot)
        print(i, spike_count, result)

        if file_tag[i] == result:
            accuracy += 1

    result = pd.DataFrame(
        {
            'file_name' : file_name,
            'file_tag' : file_tag,
            'result_tag' : result_tag
        })
    result.to_excel('result_list', encoding = 'UTF8')
    print("Total accuracy is: {}".format(accuracy/total*100))



# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

#familiarize with grad function in pytorch

import numpy as onp
import pandas as pd
import csv
import jax
import jax.numpy as np
from jax import vmap,grad
from functools import partial
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense,Relu,Flatten
#from jax import random
import random
from jax import jit
#from keras.datasets import mnist,fashion_mnist
import numpy as onp
from jax.example_libraries import optimizers
from jax.tree_util import tree_multimap


#read data
ethnic_dic=pd.read_csv('sampleID.csv',usecols=['Sample (Male/Female/Unknown)','Population(s)'])
num_rows=20000
header_line=42
#in final form get rid of num_rows and remove nrows from read.csv
data=pd.read_csv('ALL.autosomes.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz',sep='\t',
                header=header_line,nrows=num_rows)
task='MXL' #the ethnic group we want to test
removal_list=['HG00104',
 'HG00134',
 'HG00135',
 'HG00152',
 'HG00156',
 'HG00249',
 'HG00270',
 'HG00302',
 'HG00303',
 'HG00312',
 'HG00359',
 'HG00377',
 'HG01471',
 'HG02168',
 'HG02169',
 'HG02170',
 'HG02173',
 'HG02176',
 'HG02358',
 'HG02405',
 'HG02436',
 'HG03171',
 'HG03393',
 'HG03398',
 'HG03431',
 'HG03462',
 'HG03549',
 'HG04301',
 'HG04302',
 'HG04303',
 'NA18527',
 'NA18576',
 'NA18791',
 'NA18955',
 'NA19044',
 'NA19359',
 'NA19371',
 'NA19398',
 'NA20537',
 'NA20816',
 'NA20829',
 'NA20831',
 'NA20873',
 'NA20883',
 'NA21121']


#processing data
def process_dict(data):
    data=data.rename(columns={'Sample (Male/Female/Unknown)':'ID','Population(s)':'eth'})
    data['eth']=data['eth'].apply(lambda x: x.split(',')[-1][1:])
    data['ID']=data['ID'].apply(lambda x: x.split(' ')[0])
    data.index=data['ID']
    data=data.drop('ID',axis=1)
    data=data.drop('NA18498')
    return data

def process_data(data,num_causal_snps=200):
    data=data[data['ALT'].isin(['A','C','G','T'])] #select SNPs with single ALT allele
    data['INFO']=data['INFO'].apply(lambda x: float(x.split(';')[3].split('=')[-1])) #extract allele freq information
    data=data[data['INFO']>0.05] #choose SNPs with allele freq more than 0.05
    data.index=data['POS'] #set ID col as index
    data=data.drop(['ID','#CHROM','POS','REF','ALT','QUAL','FILTER','INFO','FORMAT'],axis=1) #drop columns other than individual data
    data=data.applymap(lambda x: 2 if x=='1|1' else(0 if x=='0|0' else 1)) #sets 0|0 to 0 ...
    data=data.drop(removal_list,axis=1)
    data=data.T
    causal_snps=onp.arange(0,len(data.columns),len(data.columns)//num_causal_snps)
    data=data[data.columns[causal_snps]]
    data['eth']=eth_ID['eth']
    data['ID']=data.index
    data=data.set_index(['eth','ID'])
    return data





eth_ID=process_dict(ethnic_dic)
eth=eth_ID['eth'].unique()




data1=process_data(data)

#dictionary which contains randomly generated coeffiicnets for the linear model
eth_coef={}
for i in eth:
    eth_coef[i]=onp.random.uniform(-1,1,(data1.shape[1]))

x=data1
y=data1.apply(lambda x: x@eth_coef[data1.index[0][0]],axis=1)


x_train=x.drop(task,axis=0)
y_train=y.drop(task,axis=0)
eth_train=onp.delete(eth, onp.where(eth == task))
x_test=x.loc[task].to_numpy()
y_test=y.loc[task].to_numpy()
y_test=onp.reshape(y_test,(-1,1))
eth_train=onp.delete(eth, onp.where(eth == task))


#some hyperparameters
num_features=x.shape[1] #number of causal snps
reg_weight=0.1
epochs=20000
in_shape=(-1,num_features)
ethnic_grp_min_pop=60 #min population among all subpopulatiosn
batch_size=20 #inner batch size for inner loop
#K=20 #K-shot learning
num_task_sample= 5 #number of tasks to sample to meta train
lr=0.001
rng=jax.random.PRNGKey(1)





#need ethnic_grp_pop list which contains population of each ethinc group
#training data is a list of arrays each of which corresponds to an ethinic groups
#inner batch size < ethnic_grp_min_pop
def sample_tasks(outer_batch_size, inner_batch_size):
    # Select amplitude and phase for the task
    ethnic_grp_sample=random.sample(list(eth_train), k=outer_batch_size)

    def get_batch():
        xs, ys = [], []
        for j in ethnic_grp_sample:
            indices = onp.random.randint(ethnic_grp_min_pop,size=inner_batch_size)
            x= x_train.loc[j].iloc[indices].to_numpy()
            y= y_train.loc[j].iloc[indices].to_numpy()
            xs.append(x)
            ys.append(y)
        return np.stack(xs), np.stack(ys)
    x1, y1 = get_batch()
    x2, y2 = get_batch()
    return x1, y1, x2, y2


def maml_train():
    # auxilliary functions
    def loss(params, inputs, targets):
        predictions = net_apply(params, inputs)
        for i in range(len(net_params)):
            l1_params = np.linalg.norm(net_params[i], 1)
        return np.mean((targets - predictions) ** 2) + reg_weight * np.linalg.norm(net_params[0], 1)

    def accuracy(params, inputs, targets):
        predictions = net_apply(params, inputs)
        return np.mean((targets - predictions) ** 2)

    def inner_update(p, x1, y1):
        grads = grad(loss)(p, x1, y1)
        inner_sgd_fn = lambda g, state: (state - lr * g)
        # return tree_multimap(inner_sgd_fn,grads,p)
        return [(w - lr * dw)
                for w, dw in zip(p, grads)]

    def maml_loss(p, x1, y1, x2, y2):
        p2 = inner_update(p, x1, y1)
        return loss(p2, x2, y2)

    # meta training
    net_init, net_apply = Dense(1)
    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)
    out_shape, net_params = net_init(rng, in_shape)
    opt_state = opt_init(net_params)

    # vmapped version of maml loss.
    # returns scalar for all tasks.
    def batch_maml_loss(p, x1_b, y1_b, x2_b, y2_b):
        task_losses = vmap(partial(maml_loss, p))(x1_b, y1_b, x2_b, y2_b)
        return np.mean(task_losses)

    @jit
    def step(i, opt_state, x1, y1, x2, y2):
        p = get_params(opt_state)
        g = grad(batch_maml_loss)(p, x1, y1, x2, y2)
        l = batch_maml_loss(p, x1, y1, x2, y2)
        return opt_update(i, g, opt_state), l

    np_batched_maml_loss = []

    for i in range(epochs):
        x1_b, y1_b, x2_b, y2_b = sample_tasks(num_task_sample, batch_size)
        opt_state, l = step(i, opt_state, x1_b, y1_b, x2_b, y2_b)
        np_batched_maml_loss.append(l)
        if i % 1000 == 0:
            print(i, 'maml_loss', l)
    net_params = get_params(opt_state)

    # meta testing
    # meta test; train with batch_size many examples from validation set on desired task

    # pre update prediction
    pre_predictions = vmap(partial(net_apply, net_params))(x_test)
    pre_error = loss(net_params, x_test, y_test)
    print('pre update MSE=' + str(pre_error))
    # post-update prediction
    indx = onp.random.randint(x_test.shape[0], size=batch_size)
    test_indx = onp.delete(onp.arange(x_test.shape[0]), indx)
    x1, y1 = x_test[indx], y_test[indx]
    for i in range(batch_size):
        net_params = inner_update(net_params, x1, y1)
        # print('training loss '+str(l))
        # train_accuracy= accuracy(net_params,x1,y1)
        # print('train accuracy',train_accuracy)
        # post_error= loss(net_params,x_test[test_indx],y_test[test_indx])
        # print('Post step ' + str(i)+' update test MSE='+str(post_error))

    # post_predictions = vmap(partial(net_apply, net_params))(x_test)
    maml_error = accuracy(net_params, x_test[test_indx], y_test[test_indx])
    print('Test Error on Task: MSE = ', maml_error)

    return np_batched_maml_loss,maml_error





def base_linear_model():
    def update(p, x1, y1):
        grads = grad(loss)(p, x1, y1)
        #inner_sgd_fn = lambda g, state: (state - lr * g)
        # return tree_multimap(inner_sgd_fn,grads,p)
        return [(w - lr * dw)
                for w, dw in zip(p, grads)]

    def loss(params, inputs, targets):
        predictions = basenet_apply(params, inputs)
        for i in range(len(basenet_params)):
            l1_params = np.linalg.norm(basenet_params[i], 1)
        return np.mean((targets - predictions) ** 2) + reg_weight * np.linalg.norm(basenet_params[0], 1)

    def accuracy(params, inputs, targets):
        predictions = basenet_apply(params, inputs)
        return np.mean((targets - predictions) ** 2)


    basenet_init, basenet_apply = Dense(1)
    out_shape, basenet_params = basenet_init(rng, input_shape=in_shape)
    np_batched_loss = []
    for i in range(epochs):
        basenet_params = update(basenet_params, x_train.to_numpy(), y_train.to_numpy())
        l=loss(basenet_params,x_train.to_numpy(), y_train.to_numpy())
        np_batched_loss.append(l)
        if i % 1000 == 0:
            train_loss = loss(basenet_params, x_train.to_numpy(), y_train.to_numpy())
            print(i, 'training loss', train_loss)

    indx = onp.random.randint(x_test.shape[0], size=batch_size)
    test_indx = onp.delete(onp.arange(x_test.shape[0]), indx)
    x1, y1 = x_test[indx], y_test[indx]
    for i in range(batch_size):
        basenet_params = update(basenet_params, x1, y1)

    test_error = accuracy(basenet_params, x_test, y_test)

    print('test error MSE', test_error)

    indx = onp.random.randint(x_test.shape[0], size=batch_size)
    test_indx = onp.delete(onp.arange(x_test.shape[0]), indx)
    x1, y1 = x_test[indx], y_test[indx]
    for i in range(batch_size):
        basenet_params = inner_update(basenet_params, x1, y1)

    lin_test_error = accuracy(basenet_params, x_test, y_test)

    print('test error MSE', lin_test_error)
    return np_batched_loss,lin_test_error

alpha=maml_train()
maml_error=alpha[1]

beta=base_linear_model()
lin_test_error=beta[1]

L=['maml_error ',str(maml_error)+'\n','lin_error ',str(lin_test_error)]
f=open('results.txt','w')
f.writelines(L)
f.close()

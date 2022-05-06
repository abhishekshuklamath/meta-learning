import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import random
import tqdm
import jax
import jax.numpy as jnp
from jax import vmap,grad
from functools import partial
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense,Relu,Flatten,Sigmoid
from jax import jit
from jax.example_libraries import optimizers
from jax.tree_util import tree_multimap

#hyperparameters
epochs=2000
batch_size=30 #inner batch size for inner loop#K=20 #K-shot learning
num_task_sample= 4 #number of tasks to sample to meta train
times=10
test_task_num=5
lr=0.00001
rng=jax.random.PRNGKey(1)
num_task_sample=5
ethnic_grp_min_pop=60 #min population among all subpopulatiosn
reg_weight=0.1
reg_weight_lin=0.01
reg_weight_maml=0.01


def linear_model(eth, x, hsq=0.1,num_causal_snps=10,verbose=0):
    linear_causal_snps = np.random.choice(x.columns, size=num_causal_snps,
                                               replace=False) #randomly choose num_causal_snps form the columns of x
    linear_model_coef = []  # array of 0 or 1 depending on whether a particular column is causal snp or not
    for i in x.columns:
        if i in linear_causal_snps:
            linear_model_coef.append(1)
        else:
            linear_model_coef.append(0)

    # dictionary which contains randomly generated coefficnets for the linear model
    eth_coef = {}
    eth_errors = {}
    #eth_errors_sigma2 = np.ndarray()
    #eth_errors.update([(ethnicity, errors / 100) for ethnicity, errors in zip(eth, eth_errors_sigma2)])
    #varx=np.sum(np.multiply(np.cov(x,rowvar=False),np.identity(x.shape[1])))
    #eth_errors_sigma2=-varx+ (varx/hsq)
    for i in eth:
        varx = np.sum(np.multiply(np.cov(x.loc[i], rowvar=False), np.identity(x.shape[1])))
        eth_errors[i]=-varx+ (varx/hsq)
    total = sum(eth_errors.values(), 0.0)
    eth_errors = {k: v / total for k, v in eth_errors.items()}
    for i in eth:
        eth_coef[i] = np.random.uniform(-1, 1, (x.shape[1]))
        eth_coef[i]= np.multiply(eth_coef[i],linear_model_coef)

    y = x.apply(lambda a: a @ eth_coef[a.name[0]] + np.random.normal(scale=eth_errors[a.name[0]]), axis=1)
    if verbose==0:
        return y
    if verbose==1:
        return y,eth_coef,linear_causal_snps

def hetero_model(eth, x, max_threshold=8, num_heteromodel_causal_snps=10):
    threshold_vec = np.random.randint(max_threshold,
                                      size=len(eth))  # random choice of thresholds as many as ethnicities
    heteromodel_causal_snps = np.random.choice(x.columns, size=num_heteromodel_causal_snps,
                                               replace=False)  # choose 10 causal snps from columns of x
   # heteromodel_causal_snps= hetero_causal_snps
    heteromodel_coef = []  # array of 0 or 1 depending on whether a particular column is causal snp or not
    heteromodel_thresh = {}  # dictionary of thresholds for each ethnicity
    for i in x.columns:
        if i in heteromodel_causal_snps:
            heteromodel_coef.append(1)
        else:
            heteromodel_coef.append(0)

    j = 0
    for i in eth:
        heteromodel_thresh[i] = threshold_vec[j]
        j = j + 1

    y = x.apply(lambda t: 1 if t @ heteromodel_coef > heteromodel_thresh[t.name[0]] else 0, axis=1)

    subpop_phenotype_prop=[]
    for i in eth:
        subpop_phenotype_prop.append(sum(y.loc[i] == 0) / y.loc[i].size)
    #print(np.mean(subpop_phenotype_prop), np.var(subpop_phenotype_prop))
    return y


def compensatory_model(eth, x, max_threshold1=5, max_threshold2=7, num_heteromodel_causal_snps1=5,
                       num_heteromodel_causal_snps2=5):
    threshold_vec1 = np.random.randint(max_threshold1,
                                       size=len(eth))  # random choice of thresholds as many as ethnicities
    threshold_vec2 = np.random.randint(max_threshold2,
                                       size=len(eth))  # random choice of thresholds as many as ethnicities

    heteromodel_causal_snps = np.random.choice(x.columns,
                                              size=num_heteromodel_causal_snps1 + num_heteromodel_causal_snps2,
                                             replace=False)  # choose 10 causal snps from columns of x
    heteromodel_causal_snps1 = heteromodel_causal_snps[0:num_heteromodel_causal_snps1]
    heteromodel_causal_snps2 = heteromodel_causal_snps[num_heteromodel_causal_snps1:-1]

    #heteromodel_causal_snps1 = compens_causal_snps1
    #heteromodel_causal_snps2 = compens_causal_snps2
    heteromodel_coef1 = []  # array of 0 or 1 depending on whether a particular column is causal snp or not
    heteromodel_thresh1 = {}  # dictionary of thresholds for each ethnicity
    for i in x.columns:
        if i in heteromodel_causal_snps1:
            heteromodel_coef1.append(1)
        else:
            heteromodel_coef1.append(0)

    j = 0
    for i in eth:
        heteromodel_thresh1[i] = threshold_vec1[j]
        j = j + 1

    heteromodel_coef2 = []  # array of 0 or 1 depending on whether a particular column is causal snp or not
    heteromodel_thresh2 = {}  # dictionary of thresholds for each ethnicity
    for i in x.columns:
        if i in heteromodel_causal_snps2:
            heteromodel_coef2.append(1)
        else:
            heteromodel_coef2.append(0)

    j = 0
    for i in eth:
        heteromodel_thresh2[i] = threshold_vec2[j]
        j = j + 1

    y = x.apply(lambda t: 0 if (((t @ heteromodel_coef1 > heteromodel_thresh1[t.name[0]]) and
                                 (t @ heteromodel_coef2 > heteromodel_thresh2[t.name[0]])) |
                                ((t @ heteromodel_coef1 <= heteromodel_thresh1[t.name[0]]) and
                                 (t @ heteromodel_coef2 <= heteromodel_thresh2[t.name[0]])))
    else 1, axis=1)

    subpop_phenotype_prop = []
    for i in eth:
        subpop_phenotype_prop.append(sum(y.loc[i] == 0) / y.loc[i].size)
   # print(np.mean(subpop_phenotype_prop), np.var(subpop_phenotype_prop))
    # return heteromodel_thresh1,heteromodel_thresh2,y
    return y


def sample_tasks(outer_batch_size, inner_batch_size,ethnic_grp_min_pop,eth_train,x_train,y_train):
    # Select amplitude and phase for the task
    ethnic_grp_sample=random.sample(list(eth_train), k=outer_batch_size)

    def get_batch():
        xs, ys = [], []
        for j in ethnic_grp_sample:
            indices = np.random.randint(ethnic_grp_min_pop,size=inner_batch_size)
            x= x_train.loc[j].iloc[indices].to_numpy()
            y= y_train.loc[j].iloc[indices].to_numpy()
            xs.append(x)
            ys.append(y)
        return np.stack(xs), np.stack(ys)
    x1, y1 = get_batch()
    x2, y2 = get_batch()
    return x1, y1, x2, y2


def maml_model(eth_train, x_train, y_train, x_test, y_test,epochs=20000,batch_size=20,num_task_sample= 5):
    # define model
    num_features=x_train.shape[1]
    in_shape=(-1,num_features)
    net_init, net_apply = Dense(1)
    opt_init, opt_update, get_params = optimizers.adam(step_size=lr)
    out_shape, net_params = net_init(rng, in_shape)
    opt_state = opt_init(net_params)

    # auxilliary functions
    def loss(params, inputs, targets):
        predictions = net_apply(params, inputs)
        for i in range(len(net_params)):
            l1_params = jnp.linalg.norm(net_params[i], 1)
        return jnp.mean((targets - predictions) ** 2) + reg_weight_lin * jnp.linalg.norm(net_params[0], 1)

    def mse(params, inputs, targets):
        predictions = net_apply(params, inputs)
        #print(inputs)
        #print(predictions)
        #print(targets)
        #print(jnp.mean((targets - predictions) ** 2))
        return jnp.mean((targets - predictions) ** 2)

    def rsquare(params, inputs, targets):
        yhat = net_apply(params, inputs)
        y = targets
        return scipy.stats.pearsonr(np.reshape(yhat,len(yhat)), np.reshape(y,len(y)))[0]


    def inner_update(p, x1, y1):
        grads = grad(loss)(p, x1, y1)
        inner_sgd_fn = lambda g, state: (state - lr * g)
        # return tree_multimap(inner_sgd_fn,grads,p)
        return [(w - lr * dw)
                for w, dw in zip(p, grads)]

    def maml_loss(p, x1, y1, x2, y2):
        p2 = inner_update(p, x1, y1)
        return loss(p2, x2, y2)

    # vmapped version of maml loss.
    # returns scalar for all tasks.
    def batch_maml_loss(p, x1_b, y1_b, x2_b, y2_b):
        task_losses = vmap(partial(maml_loss, p))(x1_b, y1_b, x2_b, y2_b)
        return jnp.mean(task_losses)

    @jit
    def step(i, opt_state, x1, y1, x2, y2):
        p = get_params(opt_state)
        g = grad(batch_maml_loss)(p, x1, y1, x2, y2)
        l = batch_maml_loss(p, x1, y1, x2, y2)
        return opt_update(i, g, opt_state), l

    np_batched_maml_loss = []

    for i in tqdm.tqdm(range(epochs)):
        x1_b, y1_b, x2_b, y2_b = sample_tasks(num_task_sample, batch_size, ethnic_grp_min_pop, eth_train, x_train,
                                              y_train)
        opt_state, l = step(i, opt_state, x1_b, y1_b, x2_b, y2_b)
        np_batched_maml_loss.append(l)
        #if i % 1000 == 0:
        #    print(i, 'maml_loss', l)
    net_params = get_params(opt_state)

    # meta testing
    # meta test; train with batch_size many examples from validation set on desired task

    # pre update prediction
    pre_predictions = vmap(partial(net_apply, net_params))(x_test)
    pre_error = loss(net_params, x_test, y_test)
    print('pre update loss=' + str(pre_error))
    # post-update prediction
    indx = np.random.randint(x_test.shape[0], size=batch_size)
    test_indx = np.delete(np.arange(x_test.shape[0]), indx)
    x1, y1 = x_test[indx], y_test[indx]
    for i in range(batch_size):
        net_params = inner_update(net_params, x1, y1)
        # print('training loss '+str(l))
        # train_accuracy= accuracy(net_params,x1,y1)
        # print('train accuracy',train_accuracy)
        # post_error= loss(net_params,x_test[test_indx],y_test[test_indx])
        # print('Post step ' + str(i)+' update test MSE='+str(post_error))

    # post_predictions = vmap(partial(net_apply, net_params))(x_test)
    maml_err = mse(net_params, x_test[test_indx], y_test[test_indx])
    r2= rsquare(net_params, x_test[test_indx], y_test[test_indx])
    #print('Test accuracy on Task:', maml_err)

    return np_batched_maml_loss, maml_err, r2


def base_linear_model(eth_train, x_train, y_train, x_test, y_test,epochs=20000,batch_size=20):
    num_features = x_train.shape[1]
    in_shape = (-1, num_features)

    basenet_init, basenet_apply = Dense(1)
    out_shape, basenet_params = basenet_init(rng, input_shape=in_shape)
    opt_init, opt_update, get_params = optimizers.adam(step_size=lr)
    opt_state = opt_init(basenet_params)
    #print(basenet_params)
    @jit
    def step(i, opt_state, x1, y1):
        p = get_params(opt_state)
        g = grad(batch_loss)(p, x1, y1)
        l = batch_loss(p, x1, y1)
        return opt_update(i, g, opt_state), l

    def update(p, x1, y1):
        grads = grad(loss)(p, x1, y1)
        inner_sgd_fn = lambda g, state: (state - lr * g)
        return tree_multimap(inner_sgd_fn,grads,p)
        #return [(w - lr * dw)
              #  for w, dw in zip(p, grads)]

    def loss(params, inputs, targets):
        predictions = basenet_apply(params, inputs)
        return jnp.mean((targets - predictions) ** 2) + reg_weight * jnp.linalg.norm(basenet_params[0], 1)

    def batch_loss(p,x1,y1):
        task_losses = vmap(partial(loss, p))(x1, y1)
        return jnp.mean(task_losses)

    def mse(params, inputs, targets):
        predictions = basenet_apply(params, inputs)
        #print(inputs)
        #print(params)
        #print(predictions)
        return jnp.mean((targets - predictions) ** 2)

    def rsquare(params,inputs,targets):
        yhat = basenet_apply(params, inputs)
        y=targets
        return scipy.stats.pearsonr(np.reshape(yhat,len(yhat)), np.reshape(y,len(y)))[0]


    np_batched_loss = []
    for i in tqdm.tqdm(range(epochs)):
        indices = np.random.randint(x_train.shape[0], size=batch_size*num_task_sample)
        x1,y1=x_train.iloc[indices].to_numpy(), y_train.iloc[indices].to_numpy()
        opt_state, l = step(i, opt_state, x1, y1)
        np_batched_loss.append(l)
    basenet_params = get_params(opt_state)
    #  if i % 1000 == 0:
    #     train_loss = loss(basenet_params, x_train.to_numpy(), y_train.to_numpy())
    #    print(i, 'training loss', train_loss)

    indx = np.random.randint(x_test.shape[0], size=batch_size)
    test_indx = np.delete(np.arange(x_test.shape[0]), indx)
    x1, y1 = x_test[indx], y_test[indx]
    for i in range(batch_size):
        #print(str(i), basenet_params)
        basenet_params = update(basenet_params, x1, y1)

    lin_test_error = mse(basenet_params, x_test[test_indx], y_test[test_indx])
    lin_r2 = rsquare(basenet_params, x_test[test_indx], y_test[test_indx])
    #print('test error MSE', lin_test_error)
    return np_batched_loss, lin_test_error,lin_r2


def maml_logistic_model(eth_train, x_train, y_train, x_test, y_test,epochs=20000,batch_size=20,num_task_sample= 5,reg_weight=reg_weight_maml):
    # define model
    num_features=x_train.shape[1]
    in_shape=(-1,num_features)
    net_init, net_apply = stax.serial(Dense(1),Sigmoid)
    opt_init, opt_update, get_params = optimizers.adam(step_size=lr)
    out_shape, net_params = net_init(rng, in_shape)
    opt_state = opt_init(net_params)

    # auxilliary functions
    def binary_cross_entropy(y_hat, y):
        bce = y * jnp.log(y_hat) + (1 - y) * jnp.log(1 - y_hat)
        return jnp.mean(-bce)

    def loss(params, inputs, targets):
        predictions = net_apply(params, inputs)
        #print(binary_cross_entropy(predictions,targets),jnp.linalg.norm(net_params[0][0], 1))
        return binary_cross_entropy(predictions,targets)+reg_weight * jnp.linalg.norm(net_params[0][0], 1)

    def accuracy(params, inputs, targets):
        predictions = net_apply(params, inputs)
        return jnp.mean((predictions >= 1/2) == (targets >= 1/2))

    def inner_update(p, x1, y1):
        grads = grad(loss)(p, x1, y1)
        inner_sgd_fn = lambda g, state: (state - lr * g)
        return tree_multimap(inner_sgd_fn,grads,p)
        #return [(w - lr * dw) for w, dw in zip(p, grads)]

    def maml_loss(p, x1, y1, x2, y2):
        p2 = inner_update(p, x1, y1)
        return loss(p2, x2, y2)

    # vmapped version of maml loss.
    # returns scalar for all tasks.
    def batch_maml_loss(p, x1_b, y1_b, x2_b, y2_b):
        task_losses = vmap(partial(maml_loss, p))(x1_b, y1_b, x2_b, y2_b)
        return jnp.mean(task_losses)

    @jit
    def step(i, opt_state, x1, y1, x2, y2):
        p = get_params(opt_state)
        g = grad(batch_maml_loss)(p, x1, y1, x2, y2)
        l = batch_maml_loss(p, x1, y1, x2, y2)
        return opt_update(i, g, opt_state), l

    np_batched_maml_loss = []

    for i in tqdm.tqdm(range(epochs)):
        x1_b, y1_b, x2_b, y2_b = sample_tasks(num_task_sample, batch_size, ethnic_grp_min_pop, eth_train, x_train,
                                              y_train)
        opt_state, l = step(i, opt_state, x1_b, y1_b, x2_b, y2_b)
        np_batched_maml_loss.append(l)
        #if i % 1000 == 0:
        #    print(i, 'maml_loss', l)
    net_params = get_params(opt_state)

    # meta testing
    # meta test; train with batch_size many examples from validation set on desired task

    # pre update prediction
    pre_predictions = vmap(partial(net_apply, net_params))(x_test)
    pre_error = loss(net_params, x_test, y_test)
    print('pre update loss=' + str(pre_error))
    # post-update prediction
    indx = np.random.randint(x_test.shape[0], size=batch_size)
    test_indx = np.delete(np.arange(x_test.shape[0]), indx)
    x1, y1 = x_test[indx], y_test[indx]
    for i in range(batch_size):
        net_params = inner_update(net_params, x1, y1)
        # print('training loss '+str(l))
        # train_accuracy= accuracy(net_params,x1,y1)
        # print('train accuracy',train_accuracy)
        # post_error= loss(net_params,x_test[test_indx],y_test[test_indx])
        # print('Post step ' + str(i)+' update test MSE='+str(post_error))

    # post_predictions = vmap(partial(net_apply, net_params))(x_test)
    logistic_maml_acc = accuracy(net_params, x_test[test_indx], y_test[test_indx])
    #print('Test Accuracy on Task: MSE = ', logistic_maml_acc)

    return np_batched_maml_loss, logistic_maml_acc


def base_logistic_model(eth_train, x_train, y_train, x_test, y_test,epochs=20000,batch_size=20,reg_weight=reg_weight_lin):
    num_features = x_train.shape[1]
    in_shape = (-1, num_features)

    basenet_init, basenet_apply = stax.serial(Dense(1), Sigmoid)
    out_shape, basenet_params = basenet_init(rng, input_shape=in_shape)
    opt_init, opt_update, get_params = optimizers.adam(step_size=lr)
    opt_state = opt_init(basenet_params)

    @jit
    def step(i, opt_state, x1, y1):
        p = get_params(opt_state)
        g = grad(batch_loss)(p, x1, y1)
        l = batch_loss(p, x1, y1)
        return opt_update(i, g, opt_state), l


    def binary_cross_entropy(y_hat, y):
        bce = y * jnp.log(y_hat) + (1 - y) * jnp.log(1 - y_hat)
        return jnp.mean(-bce)

    def loss(params, inputs, targets):
        predictions = basenet_apply(params, inputs)
        #print(len(basenet_params))
        #print(binary_cross_entropy(predictions, targets), jnp.linalg.norm(basenet_params[0][0], 1))
        return binary_cross_entropy(predictions, targets) + reg_weight * jnp.linalg.norm(basenet_params[0][0], 1)


    def batch_loss(p,x1,y1):
        task_losses = vmap(partial(loss, p))(x1, y1)
        return jnp.mean(task_losses)

    def accuracy(params, inputs, targets):
        predictions = basenet_apply(params, inputs)
        return jnp.mean((predictions >= 1/2) == (targets >= 1/2))

    def update(p, x1, y1):
        grads = grad(loss)(p, x1, y1)
        inner_sgd_fn = lambda g, state: (state - lr * g)
        return tree_multimap(inner_sgd_fn,grads,p)

    np_batched_loss = []
    for i in tqdm.tqdm(range(epochs)):
        indices = np.random.randint(x_train.shape[0], size=batch_size * num_task_sample)
        x1, y1 = x_train.iloc[indices].to_numpy(), y_train.iloc[indices].to_numpy()
        opt_state, l = step(i, opt_state, x1, y1)
        np_batched_loss.append(l)
    basenet_params = get_params(opt_state)
    #  if i % 1000 == 0:
    #     train_loss = loss(basenet_params, x_train.to_numpy(), y_train.to_numpy())
    #    print(i, 'training loss', train_loss)

    indx = np.random.randint(x_test.shape[0], size=batch_size)
    test_indx = np.delete(np.arange(x_test.shape[0]), indx)
    x1, y1 = x_test[indx], y_test[indx]
    for i in range(batch_size):
        basenet_params = update(basenet_params, x1, y1)

    logistic_test_acc = accuracy(basenet_params, x_test, y_test)

    #print('test error accuracy', logistic_test_acc)
    return np_batched_loss, logistic_test_acc


def task_maml(task,x,y):
    # task is the ethnic subpopulation we want to test
    # preparing training data by removing the task data from training data
    x_train=x.drop(task,axis=0)
    y_train=y.drop(task,axis=0)
    eth_train=np.delete(eth,np.where(eth == task))
    x_test=x.loc[task].to_numpy()
    y_test=y.loc[task].to_numpy()
    y_test=np.reshape(y_test,(-1,1))
   # eth_train=np.delete(eth, np.where(eth == task))

    #train and test using different models
    beta = base_linear_model(eth_train, x_train, y_train, x_test, y_test, epochs, batch_size)
    lin_test_error = beta[1]
    lin_r2=beta[2]

    alpha=maml_model(eth_train,x_train,y_train,x_test,y_test,epochs,batch_size,num_task_sample)
    maml_error=alpha[1]
    maml_r2=alpha[2]

    L = task, maml_error, lin_test_error,maml_r2, lin_r2

    return alpha[0],beta[0],L

def logistic_task_maml(task,x,y,model_type):
 # task is the ethnic subpopulation we want to test
 # preparing training data by removing the task data from training data
 x_train = x.drop(task, axis=0)
 y_train = y.drop(task, axis=0)
 eth_train = np.delete(eth, np.where(eth == task))
 x_test = x.loc[task].to_numpy()
 y_test = y.loc[task].to_numpy()
 y_test = np.reshape(y_test, (-1, 1))
 eth_train = np.delete(eth, np.where(eth == task))

 beta = base_logistic_model(eth_train, x_train, y_train, x_test, y_test, epochs, batch_size)
 lin_test_error = 1-beta[1]

 alpha = maml_logistic_model(eth_train, x_train, y_train, x_test, y_test, epochs, batch_size, num_task_sample)
 maml_error = 1-alpha[1]

 L = task, maml_error, lin_test_error
 if(model_type=='hetero'):
     #hetero_acc_vec.append(L)
     return alpha[0],beta[0],L


 if(model_type=='compensatory'):
     #compensatory_acc_vec.append(L)
     return alpha[0], beta[0],L

def execution(model_type,times,test_task_num,x,pop_dic):
    x=x
    eth=pop_dic
    error_vec=[]
    tasks= random.sample(list(eth),k=test_task_num)
    maml_loss_vec=[]
    lin_loss_vec=[]
    for t in tasks:
        for i in range(times):
            if model_type=='linear':
                y=linear_model(eth,x)
                maml_loss, lin_loss, L = task_maml(t, x, y)
            if model_type=='hetero':
                y= hetero_model(eth, x)
                maml_loss, lin_loss, L = logistic_task_maml(t, x, y, model_type='hetero')
            if model_type=='compensatory':
                y=compensatory_model(eth,x)
                maml_loss, lin_loss, L = logistic_task_maml(t, x, y, model_type='compensatory')

            maml_loss_vec.append(maml_loss)
            lin_loss_vec.append(lin_loss)
            error_vec.append(L)

    maml_loss_vec1 = pd.DataFrame(maml_loss_vec)
    maml_loss_vec1 = maml_loss_vec1.mean()
    lin_loss_vec1 = pd.DataFrame(lin_loss_vec)
    lin_loss_vec1 = lin_loss_vec1.mean()

    plt.plot(maml_loss_vec1,  label='maml')
    plt.plot(lin_loss_vec1, label='lin')
    plt.legend()
    plt.savefig(model_type+str(epochs)+'.png')

    if model_type=='linear':
        print('linear')
        L1 = pd.DataFrame(error_vec, columns=['task', 'maml_error', 'lin_error','maml_r2','lin_r2'])
        a = L1.set_index('task')
        for subpop in list(a.index.unique()):
            print(subpop)
            print(a.loc[subpop].mean())

        print((a.mean()))
    else:
        L1 = pd.DataFrame(error_vec, columns=['task', 'maml_error', 'lin_error'])
        a=L1.groupby('task')
        print(a.mean())
        print(a.mean().mean())

    return L1,lin_loss_vec1,maml_loss_vec1



#input data
data1=pd.read_csv('../genotype_data/processed_data_apr05.csv',index_col=[0,1])
eth_ID=pd.read_csv('../individual_dict/eth_ID.csv',index_col=[0])
eth=eth_ID['eth'].unique()


#L1,lin_loss_vec1,maml_loss_vec1=execution('linear',times,test_task_num,data1,eth)
#print('compensatory')
from meta_learning_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#hyperparameters
epochs=1000
batch_size=30 #inner batch size for inner loop#K=20 #K-shot learning
num_task_sample= 4 #number of tasks to sample to meta train
times=5
test_task_num=3

#input data
data1=pd.read_csv('../genotype_data/processed_data_apr05.csv',index_col=[0,1])
eth_ID=pd.read_csv('../individual_dict/eth_ID.csv',index_col=[0])
eth=eth_ID['eth'].unique()


#functions
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
            x = data1
            if model_type=='linear':
                y=linear_model(eth,data1)
                maml_loss, lin_loss, L = task_maml(t, x, y)
            if model_type=='hetero':
                y= hetero_model(eth, data1)
                maml_loss, lin_loss, L = logistic_task_maml(t, x, y, model_type='hetero')
            if model_type=='compensatory':
                y=compensatory_model(eth,data1)
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
        L1 = pd.DataFrame(error_vec, columns=['task', 'maml_error', 'lin_error','maml_r2','lin_r2'])
        a = L1.set_index('task')
        for subpop in list(a.index.unique()):
            print(subpop)
            print(a.loc[subpop].mean())
        #print(a.mean())
        print((a.mean()))
    else:
        L1 = pd.DataFrame(error_vec, columns=['task', 'maml_error', 'lin_error'])
        a=L1.groupby('task')
        print(a.mean())
        print(a.mean().mean())

    return L1,lin_loss_vec1,maml_loss_vec1


#L1,lin_loss_vec1,maml_loss_vec1=execution('linear',times,test_task_num,data1,eth)
#print('compensatory')
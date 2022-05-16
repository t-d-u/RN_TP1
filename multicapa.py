#!/usr/bin/env python3

# imports {{{
import argparse,sys
import numpy as np
import csv
import itertools as it
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# }}}

# argumentos {{{

parser=argparse.ArgumentParser(formatter_class=\
        argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--filename_datos',default='tp1_ej1_training.csv',help='el\
                    procesamiento de los datos en el script fue realizado \
                    utilizando un .csv con los targets en la primera columna')
parser.add_argument('--filename_modelo',default='weights',help='nombre del \
                    archivo que es exportado y contiene el modelo entrenado')
parser.add_argument('--S', help='Nodos por capa sin contar entrada ni salida,\
                    separados por coma, sin espacios ni []',default='5')
parser.add_argument('--lr', help='learning rate',type=float,default=0.01)
parser.add_argument('--activation', help='tanh o sigmoid',default='sigmoid')
parser.add_argument('--alfa_momento', help='entre 0 y 1',default=0,type=float)
parser.add_argument('--epocas', default=8000,type=int)
# parser.add_argument('--B',help='batch size',default='P')
args=parser.parse_args()
# }}}

# datos {{{
data = pd.read_csv(args.filename_datos,header=None)

'para obtener datos aleatorizados'
def datos(data):

    data = np.random.permutation(np.array(data))
    x = ((data)[:,1:])


    x = x.astype(float)
    x = (x-x.mean(0))/np.square(x.std(0))

    if args.activation=='sigmoid':
        z = np.array([1 if dato=='M' else 0 for dato in data[:,0:1]])

    elif args.activation=='tanh':
        z = np.array([1 if dato=='M' else -1 for dato in data[:,0:1]])

    z = z.reshape(410,1)

    x_v = x[300:]
    x_train = x[:300]

    z_v = z[300:]
    z_train = z[:300]
    return x,x_train,x_v,z_train,z_v
x,x_train,x_v,z_train,z_v = datos(data)

# }}}

# matriz de correlación{{{
# matriz_corr = pd.DataFrame(data[:,1:].astype(float)).corr()
# plt.figure()
# sns.heatmap(matriz_corr,annot=True)
# plt.show()

# }}}

# arq {{{
P = x_train.shape[0]
S = [int(i) for i in args.S.split(',')]
S.insert(0,x.shape[1])
S.append(1)
# S = [x.shape[1],20,10,5,15,1] #89%
# S = [x.shape[1],20,1]
L = len(S)

# }}}

# forward {{{

# init forward {{{

W = [np.random.normal(0,0.1,(s+1,S[index+1])) for index,s in enumerate(S) if index<len(S)-1]
W.insert(0,0)

# }}}

# funcs {{{
def bias_add( V):# {{{
    bias = -np.ones( (len(V),1) )
    return np.concatenate( (V,bias), axis=1)# }}}

def activation(x):# {{{
    sigmoid = lambda x: 1/(1+np.exp(-x))
    if args.activation=='sigmoid':
        return sigmoid(x)
    elif args.activation=='tanh':
        return np.tanh(x)
# }}}
# }}}

def forward(Xh,W,predict=False):# {{{

    Y = [np.zeros((Xh.shape[0],value+1)) if index!=len(S)-1 else \
         np.zeros((Xh.shape[0],value)) for index,value in enumerate(S)]
    Y_temp = Xh

    for k in range(1,L):
        Y[k-1][:] = bias_add(Y_temp)
        Y_temp = activation(Y[k-1]@W[k])

    Y[L-1] = Y_temp
    if predict:
        return Y[L-1]

    return Y# }}}

# }}}

# backpropagation {{{
# funcs {{{
def bias_sub( V):# {{{
    return V[:,:-1]# }}}

def d_activation(x):# {{{

    d_sigmoid = lambda x: x*(1-x)
    d_tanh = lambda x: 1-x**2

    if args.activation=='sigmoid':
        return d_sigmoid(x)
    elif args.activation=='tanh':
        return d_tanh(x)
# }}}
# }}}

# init {{{

dw = [0*w for w in W]
# dw_previo = dw
alfa = args.alfa_momento
# }}}

def backprop_momento(Y,z,W,dw_previo):# {{{

    E_output = z - Y[L-1]
    dY_output = d_activation(Y[L-1])
    D_output = E_output*dY_output
    D = D_output

    for k in range(L-1,0,-1):
        dw[k][:] = lr * (Y[k-1].T@D) + alfa*dw_previo[k]
        E = D@W[k].T
        dY = d_activation(Y[k-1])
        D = bias_sub(E*dY)

    return dw
# }}}
# }}}

# training {{{
# funcs # {{{
def adaptation(W,dw):# {{{
    for k in range(1,L):
        W[k][:] += dw[k]
    return W# }}}

def estimation(z,outputs):# {{{
    return np.mean(np.square(z-outputs))# }}}
# }}}

#init training# {{{
B = P
lr = args.lr
costo_epoca=[]
error_val=[]
t=0
cost_batch=[]
cost_epoca=[]
# }}}

# batch train{{{

# while t<300000:
while t<args.epocas:
    c = 0
    H = np.random.permutation(P)
    for batch in range(0,P,B):

        x_batch = x_train[H[ batch : batch+B ]]
        z_batch = z_train[H[ batch : batch+B ]]

        Y = forward(x_batch,W)

        # c+=estimation(z_batch,Y[L-1])
        cost_batch.append(estimation(z_batch,Y[L-1]))
        # estimation del batch

        dw = backprop_momento(Y,z_batch,W,dw)
        W = adaptation(W,dw)

    # costo_epoca.append((c)/(P/B))
    cost_epoca.append(np.mean(cost_batch))

    cost_batch=[]
    error_val.append(estimation(z_v,forward(x_v,W,True)))
    t+=1
# }}}

# }}}

# valid {{{
'''
esta función puede ser utilizada para evaluar el modelo entregado con los datos
de testeo (separando objetivos de variables independientes en "datos_eval" y
"objetivo")

'''
def evaluacion(datos_eval,pesos,objetivo):
    output_modelo_entrenado = forward(datos_eval,pesos,predict=True)
    proporcion = (objetivo==np.round(output_modelo_entrenado)).sum() / \
                       len(output_modelo_entrenado)
    print(f'proporcion correctas: {proporcion}')
validacion = evaluacion(x_v,W,z_v)
# }}}

# plot {{{
plt.figure()
# plt.plot(costo_epoca)
plt.plot(cost_epoca)
plt.plot(error_val,label='valid')
plt.xlabel('épocas')
plt.ylabel('costo')
plt.legend()
plt.show()
# }}}

# {{{ exportar modelo (lista de pesos W) a archivo filename_modelo
np.savez(f'{args.filename_modelo}.npz',W=np.array(W,dtype=object))

'''

para cargar las matrices que queden almacenadas en filename_modelo.npz luego
de entrenar un modelo nuevo:

np.load('filename_modelo.npz',allow_pickle=True)['W']


'''
# }}}

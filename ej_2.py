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

# argumentos {{{z

parser=argparse.ArgumentParser(formatter_class=\
        argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--filename_datos',default='tp1_ej2_training.csv',help=\
                    'los datos son procesados considerando un .csv con los \
                    outputs en las últimas dos columnas')
parser.add_argument('--filename_modelo',default='weights_ej_2',help='nombre \
                    del archivo exportado que contiene el modelo entrenado')
parser.add_argument('--S', help='Nodos por capa sin contar entrada ni salida,\
                    separados por coma, sin espacios ni []',default='2')
parser.add_argument('--lr', help='learning rate',type=float,default=0.01)
parser.add_argument('--activation', help='tanh o sigmoid',default='sigmoid')
parser.add_argument('--alfa_momento', help='entre 0 y 1',default=0,type=float)
parser.add_argument('--epocas', default=20,type=int)
parser.add_argument('--exportar',default=True,help='si el usuario desea \
                    exportar el modelo entrenado al archivo filename_modelo.npz')
# parser.add_argument('--B',help='batch size',default='P')
args=parser.parse_args()
# }}}

# datos {{{
data = pd.read_csv(args.filename_datos,header=None)

data = np.random.permutation(np.array(data))
def min_max_norm(datos):
    max_cols = datos.max(axis=0)
    min_cols = datos.min(axis=0)
    data = (datos - min_cols)/(max_cols - min_cols)
    return data,max_cols,min_cols
data_normalizada,max_cols,min_cols = min_max_norm(data)
#pedir las columnas sirve para usarlas para normalizar el output
#de la red con los mins y maxs del objetivo:
maximos_objetivo = max_cols[-2:]
minimos_objetivo = min_cols[-2:]
#ojo: estoy usando el mismo max y min para train y valid.

def train_valid_split(datos):
    datos_train = datos[:int(3*len(datos)/4),:]
    datos_valid = datos[int(3*len(datos)/4):,:]
    return datos_train, datos_valid
datos_train, datos_valid = train_valid_split(data_normalizada)

x_train = datos_train[:,:-2]
z_train = datos_train[:,-2:]

x_v = datos_valid[:,:-2]
z_v = datos_valid[:,-2:]
objetivo_desnormalizado = z_v*(maximos_objetivo - \
                            minimos_objetivo) + minimos_objetivo



# }}}

# arq {{{
P = x_train.shape[0]
S = [int(i) for i in args.S.split(',')]
S.insert(0,x_train.shape[1])
S.append(2)
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

    for k in range(1,L-1): #en vez de L, asi hago la ultima unidad lineal
        Y[k-1][:] = bias_add(Y_temp)
        Y_temp = activation(Y[k-1]@W[k])
    Y[L-2][:] = bias_add(Y_temp)
    Y_temp = Y[L-2]@W[L-1] #unidad lineal
    Y[L-1] = (Y_temp - minimos_objetivo)/(maximos_objetivo-minimos_objetivo)
    #output normalizado
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
    # dY_output = d_activation(Y[L-1])
    dY_output = 1 #unidad lineal
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
        output_desnormalizado = Y[L-1]*(maximos_objetivo - \
                            minimos_objetivo) + minimos_objetivo

        # cost_batch.append(estimation(z_batch,Y[L-1]))
        z_batch_desnorm = z_batch*(maximos_objetivo - \
                            minimos_objetivo) + minimos_objetivo
        cost_batch.append(estimation(z_batch_desnorm,output_desnormalizado))

        # estimation del batch

        dw = backprop_momento(Y,z_batch,W,dw)
        W = adaptation(W,dw)

    # costo_epoca.append((c)/(P/B))
    cost_epoca.append(np.mean(cost_batch))

    cost_batch=[]
    # error_val.append(estimation(z_v,forward(x_v,W,True)))
    valid_desnorm = forward(x_v,W,True)*(maximos_objetivo - \
                            minimos_objetivo) + minimos_objetivo

    error_val.append(estimation(objetivo_desnormalizado,valid_desnorm))
    # yvalid = forward(x_v,W,True)
    # print(yvalid)
    t+=1
# }}}

# }}}

# evaluacion {{{
'''
esta función puede ser utilizada para evaluar el modelo entregado con los datos
de testeo (separando objetivos de variables independientes en "datos_eval" y
"objetivo")

'''
def evaluacion(datos_eval,pesos,objetivo):
    output_modelo_entrenado = forward(datos_eval,pesos,predict=True)
    # proporcion = (objetivo==np.round(output_modelo_entrenado)).sum() / \
                       # len(output_modelo_entrenado)
    output_desnormalizado = output_modelo_entrenado*(maximos_objetivo - \
                            minimos_objetivo) + minimos_objetivo
    objetivo_desnormalizado = objetivo*(maximos_objetivo - \
                            minimos_objetivo) + minimos_objetivo

    error= estimation(objetivo_desnormalizado,output_desnormalizado)
    # error = estimation(objetivo,output_modelo_entrenado)
    return error

'''
la validación en este caso consiste en hacer un promedio de los errores
cuadrados de cada patrón. es necesario hacerlo con los datos desnormalizados.
'''
# datos_train_originales, datos_valid_originales = train_valid_split(data)
# x_v_originales = datos_valid_originales[:,:-2]
# z_v_originales = datos_valid_originales[:,-2:]
# validacion = evaluacion(x_v_originales,W,z_v_originales)

#con los datos sin normalizar:
validacion = evaluacion(x_v,W,z_v)
print(f'validacion con datos no normalizados: {validacion}')

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
if args.exportar==True:
    np.savez(f'{args.filename_modelo}.npz',W=np.array(W,dtype=object))
else:
    pass


'''

para cargar las matrices que queden almacenadas en filename_modelo.npz luego
de entrenar un modelo nuevo:

np.load('filename_modelo.npz',allow_pickle=True)['W']


'''
# }}}

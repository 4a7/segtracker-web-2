import torch as torch
import torch.tensor as tensor
import math
import random

def fx(xt):
    return (xt[0]-0.7)**2-(xt[1]-0.5)**2

def P_t(triangulo_E, T_t):
    if(triangulo_E > 0):
        return torch.exp((-triangulo_E)/T_t)
    else:
        return 1

def triangulo_x(xt):
    espacio_x = espacio[0]*0.3 
    espacio_y = espacio[1]*0.3
    print("Ex: ", espacio_x)
    print("Ey: ", espacio_y)
    x1 = xt[0] + random.uniform(-espacio_x, espacio_x)
    x2 = xt[1] + random.uniform(-espacio_y, espacio_y)
    print("x1: ", x1)
    print("x2: ", x2)
    return tensor((x1, x2))

def triangulo_E(xt):
    return torch.sub(fx(xt), fx(torch.sub(xt, triangulo_x(xt))))

def next_T(t):
    return alpha*t

# Main
espacio = tensor((4.0, 4.0))
T = 1
alpha = random.uniform(0.85, 0.96)

xt = tensor((4.0, 4.0))
ite = int(random.uniform(50, 50))

for _ in range(ite):
    tE = triangulo_E(xt) # Espacio de búsqueda
    print("TE: ", tE)
    Pt = P_t(tE, T) # Tomar la probabilidad de cambio
    print("Pt: ", Pt)
    prob = random.random() # Determinar si se hace el cambio
    print("Prob: ", prob)
    if(prob <= Pt):
        xt = triangulo_x(xt)
        T = next_T(T)
        print("T: ",  T)


print(xt)
import math

def lr_lambda_ep(epoch): 
    if epoch < 5: return 2.500 
    elif epoch < 10: return 1.500
    elif epoch < 100:
        T_max = 100
        eta_min = 0.200/1.500
        t = epoch - 10 
        cosine = 0.5 * (1 + math.cos(math.pi * t / T_max)) 
        return (eta_min + (1 - eta_min) * cosine)*0.5
    else: return 0.100

def lr_lambda_eppdl(epoch): 
    if epoch < 5: return 3.000 
    elif epoch < 10: return 1.500
    elif epoch < 300:
        T_max = 300
        eta_min = 0.200/1.500
        t = epoch - 10 
        cosine = 0.5 * (1 + math.cos(math.pi * t / T_max)) 
        return eta_min + (1 - eta_min) * cosine
    else: return 0.100
    
# pdlvar2
def lr_lambda_x0pdl(epoch): 
    if epoch == 0: return 4.000 
    elif epoch in [1,2]: return 3.000
    elif epoch in [3,4]: return 2.000 
    elif epoch in [5,6,7,8,9]: return 1.500
    elif epoch < 300: 
        T_max = 300
        eta_min = 0.100/1.500
        t = epoch - 10
        cosine = 0.5 * (1 + math.cos(math.pi * t / T_max)) 
        return eta_min + (1 - eta_min) * cosine
    else: return 0.100
    
# pdlvar
def lr_lambda_x0pdl2(epoch): 
    if epoch == 0: return 4.000 
    elif epoch in [1,2]: return 3.000
    elif epoch in [3,4]: return 2.000 
    elif epoch in [5,6,7,8,9]: return 1.000
    elif epoch < 300: 
        T_max = 300
        eta_min = 0.100/1.000
        t = epoch - 10
        cosine = 0.5 * (1 + math.cos(math.pi * t / T_max)) 
        return eta_min + (1 - eta_min) * cosine
    else: return 0.100
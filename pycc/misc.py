import numpy as np
from copy import deepcopy
from scipy.optimize import curve_fit

# Define the [0,2] Padé approximant (constant numerator, quadratic denominator)
def pade_0_2(x, a0, b1, b2):
    denominator = 1 + b1 * x + b2 * x**2
    return a0 / denominator

def pade_1_0(x, a0, a1):
    return a0 + a1 * x

def pade_0_1(x, a0, b1):
    denominator = 1 + b1 * x
    return a0 / denominator

def pade_1_1(x, a0, a1, b1):
    numerator = a0 + a1 * x
    denominator = 1 + b1 * x
    return numerator / denominator

def pade_approximant(x_data,y_data,pcc_E_correction):
    # Fit the [0,2] Padé approximant to the data (nonlinear least squares)
    params, covariance = curve_fit(pade_0_2, x_data, y_data)
    # Extract the fitted coefficients
    a0, b1, b2 = params
    # Print the fitted [0,2] Padé approximant
    print(f"Fitted [0,2] Padé approximant: P(x) = {a0:.3f} / (1 + {b1:.3f}x + {b2:.3f}x^2)")
    print(pade_0_2(1.00,a0,b1,b2))
    pcc_E_correction.update({"Pade approximant [0,2]":pade_0_2(1.00,a0,b1,b2)})

    # Now do the same for the [1,0] Pade approximant
    # Fit the [1,0] Padé approximant to the data (linear fit)
    params, covariance = curve_fit(pade_1_0, x_data, y_data)
    a0, a1 = params
    print(f"Fitted [1,0] Padé approximant: P(x) = {a0:.3f} + {a1:.3f}x")
    print(pade_1_0(1.00,a0,a1))
    pcc_E_correction.update({"Pade approximant [1,0]":pade_1_0(1.00,a0,a1)})


    ##########################################################################
    # now do [0,1] Pade
    params, covariance = curve_fit(pade_0_1, x_data, y_data)
    a0, b1 = params
    print(f"Fitted [0,1] Padé approximant: P(x) = {a0:.3f} / (1 + {b1:.3f}x)")
    print(pade_0_1(1.00,a0,b1))
    pcc_E_correction.update({"Pade approximant [0,1]":pade_0_1(1.00,a0,b1)})

    ##########################################################################
    # finally [1,1] Pade approximant
    params, covariance = curve_fit(pade_1_1, x_data, y_data)
    a0, a1, b1 = params
    print(f"Fitted [1,1] Padé approximant: P(x) = ({a0:.3f} + {a1:.3f}x) / (1 + {b1:.3f}x)")
    print(pade_1_1(1.00,a0,b1,b2))
    pcc_E_correction.update({"Pade approximant [1,1]":pade_0_2(1.00,a0,a1,b1)})
    return


def zeroT2_offDiagonal(tensor):
    o = np.shape(tensor)[0]
    v = np.shape(tensor)[2]
    diagT2 = np.zeros((o,o,v,v))
    for occ in range(o):
        for virt in range(v):
            diagT2[occ,occ,virt,virt]=tensor[occ,occ,virt,virt]
    return diagT2

def zeroT2_Diagonal(tensor):
    o = np.shape(tensor)[0]
    v = np.shape(tensor)[2]
    diagT2 = np.zeros((o,o,v,v))
    diagT2 = deepcopy(tensor)
    for occ in range(o):
        for virt in range(v):
            diagT2[occ,occ,virt,virt]= 0.0 #tensor[occ,occ,virt,virt]
    return diagT2

def print_diagT2(tensor):
    o = np.shape(tensor)[0]
    v = np.shape(tensor)[2]
    for i in range(o):
        for a in range(v):
            print(tensor[i,i,a,a])

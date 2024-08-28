import pytest
import pycc
import pyscf
from numpy import linalg as LA
import numpy as np

@pytest.mark.parametrize("Basis,Method1,Method2,Method3,Method4,Method5",[('STO-6G',
    {'atomString':'H 0. 0. 0.0; F 0.917 0. 0.0','run':{"slowSOcalc":"UCCD3","stopping_eps":10**-10}},
    {'atomString':'H 0. 0. 0.0; F 0.917 0. 0.0','run':{"slowSOcalc":"UCCD4","stopping_eps":10**-10}},
    {'atomString':'H 0. 0. 0.0; F 0.917 0. 0.0','run':{"slowSOcalc":"UCCD5","stopping_eps":10**-10}},
    {'atomString':'H 0. 0. 0.0; F 0.917 0. 0.0','run':{"slowSOcalc":"UCCSD4","stopping_eps":10**-10}},
    {'atomString':'H 0. 0. 0.0; F 0.917 0. 0.0','run':{"slowSOcalc":"UCCSD5","stopping_eps":10**-10}},
    ),])


def test_ccdTypesCheck(Basis,Method1,Method2,Method3,Method4,Method5):
    Methods=[Method1,Method2,Method3,Method4,Method5]

    run_times=5
    for i in range(5):
        method_DM=[] # store the DM from subsequent runs
        for j in range(run_times): # verify consistency between separate runs
            method=Methods[i]
            atomString=method['atomString'] #f'H 0. 0. 0.0; F 0.917 0. 0.0'
            value=[]
            basis=Basis
            mol = pyscf.M(
                atom=atomString,
                verbose=5,
                symmetry =True,
                basis=basis)
            mf = mol.RHF(mol)
            
            mf.run()
            
            a = method['run']
            obj=pycc.DriveCC(mf,mol,a)
            
            
            print(obj.integralInfo.keys())
            obj.__dict__
            
            print(repr(obj))
            #help(obj)
            
            
            obj.kernel(a)
    
            obj.drive_rdm(mol,mf)
    
            totalDM = obj.rdm1["alpha"]+obj.rdm1["beta"]
    
            assert np.allclose(totalDM.T,totalDM, rtol=10E-8,atol=10E-10)
            method_DM.append(totalDM)
        check_sequential1RDMS(method_DM,run_times)


def check_sequential1RDMS(method_DM,run_times):
    for i in range(run_times):
        for j in range(i,run_times):
            assert np.allclose(method_DM[i],method_DM[j],rtol=10E-8,atol=10E-10)


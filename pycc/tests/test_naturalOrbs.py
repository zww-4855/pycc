import pytest
import pycc
import pyscf
from numpy import linalg as LA
import numpy as np

@pytest.mark.parametrize("Basis,Method1,Method2",[('DZ',
    {'atomString':'H 0. 0. 0.0; F 0.917 0. 0.0','run':{"slowSOcalc":"UCCD3","stopping_eps":10**-10}},
    {'atomString':'F 0. 0. 0.0; F 1.4119 0. 0.0','run':{"slowSOcalc":"UCCD3","stopping_eps":10**-10,}},
    ),])
def test_ccdTypesCheck(Basis,Method1,Method2):
    Methods=[Method1,Method2]

    hf_no=[1.99981,  1.99160,  1.98371,  1.98371,  1.97136,  0.02714,  0.01568, 0.01568,0.00930  ,0.00162 , 0.00024,0.00015]
    f2_no=[  1.99981,  1.99981,  1.99270,  1.98997,  1.98801,  1.98801,  1.97840,  1.97840,
  1.86421,  0.14356,  0.01634,  0.01634,  0.01080,  0.01080,  0.00883,  0.00707,
  0.00526,  0.00136,  0.00016 , 0.00016]


    for i in range(2):
        for j in range(15): # verify consistency between separate runs
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
            pycc_eigenvalues,eigenvectors=LA.eig(totalDM)
            pycc_eigenvalues=np.sort(pycc_eigenvalues)[::-1]
    
            #verify the natural orbitals I get are the same as ACES
            if i==0:
                print(pycc_eigenvalues,hf_no)
                print(np.allclose(pycc_eigenvalues,hf_no, rtol=10E-4,atol=10E-4))
                assert np.allclose(pycc_eigenvalues,hf_no, rtol=10E-4,atol=10E-4)
            else:
                print(pycc_eigenvalues,f2_no)
                print(np.allclose(pycc_eigenvalues,f2_no, rtol=10E-4,atol=10E-4))
                assert np.allclose(pycc_eigenvalues,f2_no, rtol=10E-4,atol=10E-4)
            # verify the occupation number of natural orbitals == total # of electrons in the system
            assert sum(mf.mo_occ)-sum(pycc_eigenvalues) < 10E-9


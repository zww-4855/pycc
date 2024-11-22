import pytest
import pycc
import pyscf
from numpy import linalg as LA
import numpy as np

#@pytest.mark.parametrize("Basis,Method1,Method2,Method3,Method4,Method5",[('STO-6G',
#    {'atomString':'H 0. 0. 0.0; F 0.917 0. 0.0','run':{"slowSOcalc":"UCCD3","stopping_eps":10**-10}},
#    {'atomString':'H 0. 0. 0.0; F 0.917 0. 0.0','run':{"slowSOcalc":"UCCD4","stopping_eps":10**-10}},
#    {'atomString':'H 0. 0. 0.0; F 0.917 0. 0.0','run':{"slowSOcalc":"UCCD5","stopping_eps":10**-10}},
#    {'atomString':'H 0. 0. 0.0; F 0.917 0. 0.0','run':{"slowSOcalc":"UCCSD4","stopping_eps":10**-10}},
#    {'atomString':'H 0. 0. 0.0; F 0.917 0. 0.0','run':{"slowSOcalc":"UCCSD5","stopping_eps":10**-10}},
#    ),])
@pytest.mark.parametrize("Basis,Method,Answer",[
    ('sto-6g',{"slowSOcalc":"UCCD3"},-0.026114434347),
    ('lanl2dz',{"slowSOcalc":"UCCD3"},-0.121756382246),
    ('sto-6g',{"slowSOcalc":"UCCD4"}, -0.025726821959),
    ('lanl2dz',{"slowSOcalc":"UCCD4"},-0.122140069390),
    ('sto-6g',{"slowSOcalc":"UCCD5"},-0.025631309603),
    ('lanl2dz',{"slowSOcalc":"UCCD5"},-0.122453830814),
    ('lanl2dz',{"slowSOcalc":"UCCSD4"},-0.123567071975 ),
    ('lanl2dz',{"slowSOcalc":"UCCSD5"}, -0.123039552600),
    ('sto-6g',{"slowSOcalc":"UCCSD4"}, -0.025905863413),
    ('sto-6g',{"slowSOcalc":"UCCSD5"}, -0.025860329809)]
    )
def test_uccn_reliability(Basis,Method,Answer):
#def test_ccdTypesCheck(Basis,Method1,Method2,Method3,Method4,Method5):
#    Methods=[Method1,Method2,Method3,Method4,Method5]

    atomString=f'H 0. 0. 0.0; F 0.917 0. 0.0'
    value=[]
    mol = pyscf.M(
        atom=atomString,
        verbose=5,
        symmetry =True,
        basis=Basis)
    mf = mol.RHF(mol) 
    mf.run()
    
    tmprun=Method
    tmprun.update({"stopping_eps":10**-10,"diis_size":3,"diis_start_cycle":2,"max_iter":100})
    obj=pycc.DriveCC(mf,mol,tmprun)
    
    obj.kernel(tmprun)

    totalE = obj.correlationE["totalE"]
    corrE  = obj.correlationE["totalCorrE"]
    diff = abs(Answer - corrE)
    assert (diff <= 10**-8)

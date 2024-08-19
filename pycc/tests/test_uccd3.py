import pytest
import pycc
import pyscf

@pytest.mark.parametrize("Basis,Method1,Method2,Method3,Method4",[('STO-6G',
    {'atomString':'H 0. 0. 0.0; F 0.917 0. 0.0','run':{"slowSOcalc":"UCCD3","stopping_eps":10**-10,"diis_size":5,"diis_start_cycle":2}},
    {'atomString':'H 0. 0. 0.0; F 0.917 0. 0.0','run':{"slowSOcalc":"UCCD3","stopping_eps":10**-10,"diis_size":5,"diis_start_cycle":2,"dropcore":1}},
    {'atomString':'F 0. 0. 0.0; F 1.4119 0. 0.0','run':{"slowSOcalc":"UCCD3","stopping_eps":10**-10,"diis_size":5,"diis_start_cycle":2}},
    {'atomString':'F 0. 0. 0.0; F 1.4119 0. 0.0','run':{"slowSOcalc":"UCCD3","stopping_eps":10**-10,"diis_size":5,"diis_start_cycle":2,"dropcore":2}}
    ),])
def test_ccdTypesCheck(Basis,Method1,Method2,Method3,Method4):
    cfour_lccd_correlationE=[-0.02611445478075323,-0.02609405823112638,-0.08886948254497934,-0.08883325578465168]
    pyccE=[]
    Methods=[Method1,Method2,Method3,Method4]
    for i in range(4):
        method=Methods[i]
        atomString=method['atomString'] #f'H 0. 0. 0.0; F 0.917 0. 0.0'
        value=[]
        basis=Basis
        mol = pyscf.M(
            atom=atomString,
            verbose=5,
            symmetry =True,
            basis=basis)
        mf = mol.UHF(mol)
        
        mf.run()
        
        a = method['run']
        obj=pycc.DriveCC(mf,mol,a)
        
        
        print(obj.integralInfo.keys())
        obj.__dict__
        
        print(repr(obj))
        #help(obj)
        
        
        obj.kernel(a)

        correlationE = obj.correlationE["totalCorrE"]
        pyccE.append(correlationE)
        assert (abs((abs(correlationE) - abs(cfour_lccd_correlationE[i]))) < 10**-7)
    print('correlation energies:',pyccE)

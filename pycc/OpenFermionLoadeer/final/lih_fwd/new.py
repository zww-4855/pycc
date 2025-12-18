import numpy as np
from pyscf import gto, scf
from openfermion import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion import InteractionOperator
from openfermion.chem.molecular_data import antisymtei
from openfermion.chem import molecular_data
from openfermionpyscf import run_pyscf
from openfermion.chem import MolecularData

import time
from openfermion.ops import InteractionOperator, FermionOperator
import scipy.linalg as la
from scipy.optimize import minimize

from openfermionpyscf import generate_molecular_hamiltonian
import pycc.OpenFermionLoadeer.pycc_classes as pycc


geometry = [
    ['H', [0,0,0]],
    ['Li', [0,0,2.5]]
]
basis ='STO-3G'
multiplicity = 0  
charge = 0

pyscf_mol, pyscf_mf = pycc.initialize_pyscf(geometry, basis,multiplicity, charge,multiplicity)
cc_info = {"dropcore":0}   #generate_molecular_hamiltonian(geometry,basis,multiplicity, charge)
print(pyscf_mf.mo_coeff)
obj2 = pycc.MeanFieldToJWspin(pyscf_mol,pyscf_mf,cc_info)
obj2.collect_data(pyscf_mol,pyscf_mf,cc_info)

H = obj2.Hdef


from scipy.sparse import linalg
## Compute ground energy
eigs, _ = linalg.eigsh(H, k=1, which="SA")
ground_energy = eigs[0]

print("e cmp:",ground_energy)



psi0 = obj2.psi0
E1100 = psi0 @ H @ psi0 
print("<1100|H|1100> = {}\n".format(E1100))

print('nelectrons:',obj2.n_electrons)
obj2.drive_vqe()

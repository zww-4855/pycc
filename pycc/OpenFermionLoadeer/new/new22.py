import numpy as np
from pyscf import gto, scf
from openfermion import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion import InteractionOperator
from openfermion.chem.molecular_data import antisymtei
from openfermion.chem import molecular_data
from openfermionpyscf import run_pyscf
from openfermion.chem import MolecularData

## https://quantumcomputing.stackexchange.com/questions/23801/hartree-fock-state-in-openfermion
geometry = [
    ['H', [0,0,0]],
    ['H', [0,0,1]]
]
basis = 'sto-3g'
multiplicity = 1
charge = 0

molecule = MolecularData(geometry, basis, multiplicity, charge)
molecule = run_pyscf(molecule, run_scf=True)
HF_energy = molecule.hf_energy

#convert Hamiltonian to matrix form; this is a 4 qubit system
H = get_sparse_operator(molecule.get_molecular_hamiltonian())
n_qubits = 4

#Now, compute energy manually with formula E_psi = <psi|H|psi>

#compute energy with |1100>. This is a 2**n_qubits sized vector with 1 in the int('1100', 2) index
v1100 = np.zeros(2**n_qubits)
v1100[int('1100', 2)] = 1
E1100 = v1100 @ H @ v1100

#compute energy with |0011>. This is a 2**n_qubits sized vector with 1 in the int('0011', 2) index
v0011 = np.zeros(2**n_qubits)
v0011[int('0011', 2)] = 1
E0011 = v0011 @ H @ v0011

#print results
print("Hartree Fock Energy: {}\n".format(HF_energy))
print("<1100|H|1100> = {}\n".format(E1100))
print("<0011|H|0011> = {}\n".format(E0011))



##
#https://github.com/quantumlib/OpenFermion/blob/v1.7.1/src/openfermion/chem/molecular_data.py
import openfermion
one_body_integrals, two_body_integrals = molecule.get_integrals()
one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
            one_body_integrals, two_body_integrals
        )

# Cast to InteractionOperator class and return.
molecular_hamiltonian = InteractionOperator(
            molecule.nuclear_repulsion, one_body_coefficients, 1 / 2 * two_body_coefficients
        )
#oei = molecule.one_body_integrals
#tei = molecule.two_body_integrals
#sys.exit()
#hamiltonian = InteractionOperator(E_nuc, h1s, h2s)
#Hprime =  get_sparse_operator(molecular_hamiltonian) 
Hprime = get_fermion_operator(molecular_hamiltonian)
Hprime = get_sparse_operator(Hprime)
#Hprime = openfermion.linalg.jordan_wigner_sparse(Hprime)

### https://github.com/quantumlib/OpenFermion/blob/v1.7.1/src/openfermion/linalg/sparse_tools.py
hf_state = openfermion.linalg.jw_hartree_fock_state(2, 4)
testE = openfermion.linalg.expectation_computational_basis_state(Hprime,[1,1,0,0])
print('test',testE)



## 1) Build molecule and run SCF
#mol = gto.Mole()
#mol.build(atom='H 0 0 0; H 0 0 0.8', basis='sto-3g', spin=0, charge=0)
#
#mf = scf.RHF(mol)
#mf.kernel()
#
## 2) Get MO integrals
#C = mf.mo_coeff
#h1e_ao = mf.get_hcore()
#h1e = C.T @ h1e_ao @ C  # 1-electron integrals in MO basis
#
#eri_ao = mol.intor("int2e")  # 2-electron AO integrals
#h2e = np.einsum("pi,qj,rk,sl,ijkl->pqrs", C, C, C, C, eri_ao, optimize=True)
#
#
#
#geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.73))]
#basis = "sto-6g"
#multiplicity = 1
#charge = 0
#
#from openfermion.chem import MolecularData
#
#molecule = MolecularData(geometry, basis, multiplicity)
#
## Run PySCF and attach results to the MolecularData object
#molecule = run_pyscf(molecule, run_scf=True)
#print(molecule.hf_energy)
#
#oei_mo,  tei_mo = molecule.one_body_integrals, molecule.two_body_integrals
## 3) Convert to spin-orbital integrals
##h1s, h2s = spinorb_from_spatial(oei_mo,  tei_mo)
#h2s = antisymtei(tei_mo)
#h1s, h2s = spinorb_from_spatial(oei_mo,  h2s)
##H = molecule
##oei , tei = molecule.get_integrals()
#
##sys.exit()
## 4) Build InteractionOperator
#E_nuc = mol.energy_nuc()
#hamiltonian = InteractionOperator(E_nuc, h1s, h2s)
#
## 5) Convert to FermionOperator and QubitOperator
#ferm_op = get_fermion_operator(hamiltonian)
#qubit_op = jordan_wigner(ferm_op)
#
## 6) Sparse matrix representation
#H_sparse = get_sparse_operator(qubit_op)
#print("Sparse Hamiltonian shape:", H_sparse.shape)
#
## 7) HF energy check
#n_spin_orb = h1s.shape[0]
#hf_occ = [1] * mol.nelectron + [0] * (n_spin_orb - mol.nelectron)
#hf_index = int("".join(str(b) for b in hf_occ[::-1]), 2)
#
#psi = np.zeros(2 ** n_spin_orb, dtype=complex)
##psi[hf_index] = 1.0
##print("psi:",psi,hf_occ)
#psi[0]=1
#psi[8]=1
#import openfermion
#psi2 = openfermion.linalg.jw_hartree_fock_state(2, 4)
#print("psi 2:",psi2)
#from scipy.sparse import linalg
## Compute ground energy
#eigs, _ = linalg.eigsh(H_sparse, k=1, which="SA")
#ground_energy = eigs[0]
#
#print("Ground_energy: {}".format(ground_energy))
#
#E_HF = np.vdot(psi2, H_sparse @ psi2).real
#print("HF Energy from OpenFermion Hamiltonian:", E_HF)
#print("SCF Energy from PySCF:", mf.e_tot)


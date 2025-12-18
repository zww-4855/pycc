import numpy as np
from pyscf import gto, scf
from openfermion import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion import InteractionOperator
from openfermion.chem.molecular_data import antisymtei
from openfermion.chem import molecular_data

# 1) Build molecule and run SCF
mol = gto.Mole()
mol.build(atom='H 0 0 0; H 0 0 0.8', basis='sto-3g', spin=0, charge=0)

mf = scf.RHF(mol)
mf.kernel()

# 2) Get MO integrals
C = mf.mo_coeff
h1e_ao = mf.get_hcore()
h1e = C.T @ h1e_ao @ C  # 1-electron integrals in MO basis

eri_ao = mol.intor("int2e")  # 2-electron AO integrals
h2e = np.einsum("pi,qj,rk,sl,ijkl->pqrs", C, C, C, C, eri_ao, optimize=True)
#h2e = h2e.transpose(0,1,3,2)
# 3) Convert to spin-orbital integrals
h1s, h2s = spinorb_from_spatial(h1e, h2e)
#h2s = antisymtei(h2s)
oei , tei = molecular_data.get_integrals()

sys.exit()
# 4) Build InteractionOperator
E_nuc = mol.energy_nuc()
hamiltonian = InteractionOperator(E_nuc, h1s, h2s)

# 5) Convert to FermionOperator and QubitOperator
ferm_op = get_fermion_operator(hamiltonian)
qubit_op = jordan_wigner(ferm_op)

# 6) Sparse matrix representation
H_sparse = get_sparse_operator(qubit_op)
print("Sparse Hamiltonian shape:", H_sparse.shape)

# 7) HF energy check
n_spin_orb = h1s.shape[0]
hf_occ = [1] * mol.nelectron + [0] * (n_spin_orb - mol.nelectron)
hf_index = int("".join(str(b) for b in hf_occ[::-1]), 2)

psi = np.zeros(2 ** n_spin_orb, dtype=complex)
#psi[hf_index] = 1.0
#print("psi:",psi,hf_occ)
psi[0]=1
psi[8]=1
import openfermion
psi2 = openfermion.linalg.jw_hartree_fock_state(2, 4)
print("psi 2:",psi2)
from scipy.sparse import linalg
# Compute ground energy
eigs, _ = linalg.eigsh(H_sparse, k=1, which="SA")
ground_energy = eigs[0]

print("Ground_energy: {}".format(ground_energy))

E_HF = np.vdot(psi2, H_sparse @ psi2).real
print("HF Energy from OpenFermion Hamiltonian:", E_HF)
print("SCF Energy from PySCF:", mf.e_tot)


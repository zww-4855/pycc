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
## https://quantumcomputing.stackexchange.com/questions/23801/hartree-fock-state-in-openfermion
def run_pyscf2():
    geometry = [
        ['H', [0,0,0]],
        ['Li', [0,0,1]]
    ]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule = run_pyscf(molecule, run_scf=True)
    HF_energy = molecule.hf_energy
    return molecule


molecule = run_pyscf2()
def build_H_way1(molecule):
    H = get_sparse_operator(molecule.get_molecular_hamiltonian())
    return H

def build_H_way2(molecule):
    one_body_integrals, two_body_integrals = molecule.get_integrals()
    one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
                one_body_integrals, two_body_integrals
            )

    print('two body coeff:',two_body_coefficients,type(two_body_coefficients),np.shape(two_body_coefficients))
    mol = InteractionOperator(molecule.nuclear_repulsion, one_body_coefficients, two_body_coefficients* 0.5 )
    ham_fop = get_fermion_operator(mol)
    ham_mat = get_sparse_operator(jordan_wigner(ham_fop))
    return ham_mat


HF_energy = molecule.hf_energy
H=build_H_way1(molecule)
H1=build_H_way2(molecule)



########################################################################################
########################################################################################
########################################################################################

import pycc.OpenFermionLoadeer.pycc_classes as pycc
def run_pyscf():
    mol = gto.Mole()
    mol.atom = [
        ["H", (0.0, 0.0, 0.0)],
        ["Li", (0.0, 0.0, 1)],
    ]
    mol.basis = "sto-3g"
    mol.spin = 0
    mol.charge = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    print(dir(mol))
    # electron / orbital counts
    print(mol.nelec)
    return mol, mf

pyscf_mol, pyscf_mf = run_pyscf()
cc_info = {"dropcore":0}

spats = pycc.SpatialOrbInfo(pyscf_mol,pyscf_mf,cc_info)
obj2 = pycc.MeanFieldToJWspin(pyscf_mol,pyscf_mf,cc_info,spats)
obj2.collect_data(pyscf_mol,pyscf_mf,cc_info)
HHH = obj2.export_FermionOperator()
obj = obj2

#obj = generate_mf_data()

print(type(obj.g),np.shape(obj.g),np.shape(obj.hcore),obj.E_nuc)
interaction = InteractionOperator(constant=obj.E_nuc, one_body_tensor=obj.int_H, two_body_tensor=0.5*obj.int_V)

# FermionOperator and ensure Hermitian
fermion_ham = get_fermion_operator(interaction)
#fermion_ham = 0.5 * (fermion_ham + hermitian_conjugated(fermion_ham))

# Map to qubit operator (Jordan-Wigner)
#qubit_ham = jordan_wigner(fermion_ham)

# Convert to sparse matrix (scipy.sparse) and then to dense for small system
H_sparse = get_sparse_operator(jordan_wigner(fermion_ham))   # sparse (2^n x 2^n)
##H = H_sparse.toarray()                                # dense matrix (only OK for small n_qubits)

#print("original H:", np.shape(0.5*obj.int_V),obj.int_H)
#print("now H:",np.shape(HHH),HHH)




# Convert to sparse matrix
#sparse_op = get_sparse_operator(obj.int_Vspin_fermiop, obj.n_spin_orbitals)

# Convert sparse to dense NumPy array
#matrix = sparse_op.toarray()
#print("final spin orb g tensor:",matrix)
print("compare against g",obj.g)
print("sparseop:",obj.int_Vspin_fermiop)

sys.exit()

########################################################################################
########################################################################################

from scipy.sparse import linalg
## Compute ground energy
eigs, _ = linalg.eigsh(H, k=1, which="SA")
ground_energy = eigs[0]

eigs, _ = linalg.eigsh(H1, k=1, which="SA")
groundE = eigs[0]

print("e cmp:",ground_energy,groundE)

#sys.exit()


print("Orb energies:",molecule.orbital_energies)


n_qubits = 4

#Now, compute energy manually with formula E_psi = <psi|H|psi>

#v1100 = build_hf_state(obj) #np.zeros(2**n_qubits)


v1100 = obj.build_hf_state()
#v1100[int('1100', 2)] = 1
print('v1100:', v1100)
#sys.exit()
E1100 = v1100 @ H @ v1100

#compute energy with |0011>. This is a 2**n_qubits sized vector with 1 in the int('0011', 2) index
#v0011 = np.zeros(2**n_qubits)
#v0011[int('0011', 2)] = 1
#E0011 = v0011 @ H @ v0011

#print results
print("Hartree Fock Energy: {}\n".format(HF_energy))
print("<1100|H|1100> = {}\n".format(E1100))
#print("<0011|H|0011> = {}\n".format(E0011))

sys.exit()

# build uccsd generator
n_occ_spin = molecule.n_electrons
occ_spin_idxs = list(range(n_occ_spin))
virt_spin_idxs = list(range(n_occ_spin, molecule.n_qubits))
singles = []   # each element: (p, q) with p in virt, q in occ -> excitation a_p^\dag a_q
doubles = []   # each element: (p, q, r, s) with p,q in virt; r,s in occ -> a_p^\dag a_q^\dag a_r a_s

def generate_singles(occ_spin_idxs,virt_spin_idxs):
    singles = []
    for q in occ_spin_idxs:
        for p in virt_spin_idxs:
            singles.append((p, q))

    return singles

def generate_doubles(occ_spin_idxs,virt_spin_idxs):
    doubles = []
    for r_idx in range(len(occ_spin_idxs)):
        for s_idx in range(r_idx + 1, len(occ_spin_idxs)):
            r = occ_spin_idxs[r_idx]
            s = occ_spin_idxs[s_idx]
            for p_idx in range(len(virt_spin_idxs)):
                for q_idx in range(p_idx + 1, len(virt_spin_idxs)):
                    p = virt_spin_idxs[p_idx]
                    q_ = virt_spin_idxs[q_idx]
                    doubles.append((p, q_, r, s))
    return doubles

singles = generate_singles(occ_spin_idxs,virt_spin_idxs)
doubles = generate_doubles(occ_spin_idxs,virt_spin_idxs)
n_singles = len(singles)
n_doubles = len(doubles)
n_params = n_singles + n_doubles
print(f"n_singles={n_singles}, n_doubles={n_doubles}, total parameters={n_params}")

# Helper to build FermionOperator for a single excitation E_pq = a_p^\dag a_q
def single_excitation_op(p, q):
    # returns FermionOperator representing (a_p^\dag a_q)
    return FermionOperator(((p, 1), (q, 0)), 1.0)

# Helper for double excitation E_pqrs = a_p^\dag a_q^\dag a_r a_s
def double_excitation_op(p, q, r, s):
    return FermionOperator(((p, 1), (q, 1), (r, 0), (s, 0)), 1.0)

# Build list of FermionOperators for generator terms
gen_fermion_terms = []
# singles
for (p, q) in singles:
    gen_fermion_terms.append(single_excitation_op(p, q))
# doubles
for (p, q, r, s) in doubles:
    gen_fermion_terms.append(double_excitation_op(p, q, r, s))


# Convert each fermion term to qubit operator (JW)
gen_qubit_terms = [jordan_wigner(term) for term in gen_fermion_terms]

# Convert to dense matrices (qubit space)
gen_unitary_mats = []
for qop in gen_qubit_terms:
    # Ensure hermiticity handling: generator element is (T_i - T_i^\dag)
    # We'll store the qubit operator for T_i (not yet anti-Hermitian)
    mat = get_sparse_operator(qop, molecule.n_qubits).toarray()
    gen_unitary_mats.append(mat)

# -------------------------
# 5) Build trotterized U(theta) action
# -------------------------
def trotterized_u_of_theta(theta, reps=1):
    """
    Build dense unitary matrix U(theta) = Prod_{rep} Prod_i exp( (theta_i/ reps) * (G_i - G_i^\dag) )
    where G_i are fermionic excitation qubit-mapped matrices (here gen_unitary_mats[i]).
    Note: ordering is fixed as the generation order above.
    Returns dense matrix (2^n x 2^n).
    """
    U = np.eye(2**molecule.n_qubits, dtype=complex)
    # For each T_i construct A_i = G_i (matrix), then anti-hermitian generator = (A_i - A_i^\dag)
    for _rep in range(reps):
        for i, A in enumerate(gen_unitary_mats):
            theta_i = theta[i]
            G = A
            antiH = G - G.conj().T
            # exponentiate: exp( (theta_i / reps) * antiH )
            mat = la.expm((theta_i / reps) * antiH)
            U = mat.dot(U)   # left-multiply: apply this small-unitary next
    return U

# -------------------------
# 6) Energy evaluation and cost function
# -------------------------
H_dense = H
psi0= v1100

def transform_theta(theta):
    theta0 = InteractionOperator(0.0, 0.0, theta )
    theta0_fop = get_fermion_operator(theta0)
    theta0_mat = get_sparse_operator(jordan_wigner(theta0_fop))
    print("transformed theta:",np.shape(theta0_mat),theta0_mat,theta)
    return theta0_mat

def energy_from_theta(theta, reps=1):
    """
    Compute energy expectation <psi(theta)| H |psi(theta)> where
    |psi(theta)> = U(theta) |HF>.
    """
    U = trotterized_u_of_theta(theta, reps=reps)
    psi = U.dot(psi0)

    #[T] correction
#    thetaT = transform_theta(theta)
    ## compute [T] fourth order diagrams-> sqrBrakT_scale
    # expectation value
    E = np.vdot(psi, H_dense.dot(psi)).real # + sqrBrakT_scale
    return E

# Quick test: zero parameters should give HF energy (within numerical error)
theta0 = np.zeros(n_params)
E0 = energy_from_theta(theta0, reps=1)
print(f"Energy at theta=0 (should be HF energy): {E0:.12f}  PySCF RHF energy: {molecule.hf_energy:.12f},{energy_from_theta(theta0):.12f}")

# -------------------------
# 7) Optimize parameters
# -------------------------
# Use a modest optimizer; for stability use BFGS or Nelder-Mead for small problems
opts = {"maxiter": 200, "disp": True}

def callback(xk):
    e = energy_from_theta(xk, reps=1)
    print(f"callback: energy={e:.12f}")

print("Starting optimization... (this may take some time for larger ansatz sizes)")
t_start = time.time()
res = minimize(energy_from_theta, x0=theta0, method="BFGS", options=opts, callback=callback)
t_end = time.time()
print("Optimization finished in %.2f s" % (t_end - t_start))
print("Success:", res.success)
print("Final energy:", res.fun)
print("Final parameters (first 10):", res.x[:10])

# -------------------------
# 8) Compare with FCI (if small)
# -------------------------
try:
    from pyscf import fci
    cisolver = fci.FCI(molecule)
    e_fci = cisolver.kernel()[0]
    print(f"FCI energy: {e_fci:.12f}")
except Exception as e:
    print("FCI comparison skipped:", e)




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
testE = openfermion.linalg.expectation_computational_basis_state(H,hf_state)
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


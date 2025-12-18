
#try:
#    from pycc.OpenFermionLoadeer import test_vqe 
#except ImportError:
#    from .pycc.OpenFermionLoadeer import test_vqe


"""
UCCSD (Trotterized) with OpenFermion + PySCF only.
- Builds molecule (PySCF)
- Constructs InteractionOperator -> FermionOperator (OpenFermion)
- Generates standard singles/doubles UCCSD generator
- Trotterizes the unitary eT - Tdagger} by sequential exponentials of generators
- Applies unitary to HF reference and evaluates energy
- Optimizes parameters with scipy.optimize
WARNING: This code constructs explicit state vectors and dense matrices (2^n_qubits).
         Only practical for small systems (n_qubits <= ~16 recommended).
"""

import numpy as np
from math import comb
from functools import reduce
import time
import sys
# PySCF
from pyscf import gto, scf, ao2mo

# OpenFermion
from openfermion.ops import InteractionOperator, FermionOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.utils import hermitian_conjugated
from openfermion.linalg import get_sparse_operator  # returns a scipy sparse operator
from openfermionpyscf import run_pyscf
import openfermion

# SciPy / NumPy for exponentiation and optimization
import scipy.linalg as la
from scipy.optimize import minimize

import copy as cp
from pyscf import gto, scf, mcscf, fci, ao2mo,  cc
from pyscf.cc import ccsd
import pyscf
from pyscf import lib

# -------------------------
# 1) Molecule with PySCF
# -------------------------
# Example: H2 in STO-3G
def run_pyscf():
    mol = gto.Mole()
    mol.atom = [
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.7414)],
    ]
    mol.basis = "sto-6g"
    mol.spin = 0
    mol.charge = 0
    mol.build()
    
    mf = scf.RHF(mol)
    mf.kernel()
    print(dir(mol))
    # electron / orbital counts
    print(mol.nelec)
    return mol, mf

#import pycc
#
#from pathlib import Path
#
## Add the current directory to the Python path
#sys.path.insert(0, str(Path(__file__).parent))
#try:
#    from pycc.OpenFermionLoadeer.test_vqe import MeanFieldData
#except:
#    from .pycc.OpenFermionLoadeer.test_vqe import MeanFieldData
#obj = MeanFieldData(mf, mol,{"slowSOcalc":"CCSD"})
#
#print(dir(obj))
#
#print(obj.eps)


from pyscf import gto, scf, mcscf, fci, ao2mo,  cc
from pyscf.cc import ccsd
import pyscf
from pyscf import lib

class SpatialOrbInfo():
    def __init__(self,pyscf_mol,pyscf_mf,cc_info):
        print(type(pyscf_mol),type(pyscf_mf))
        self.n_mo =  np.shape(pyscf_mf.mo_coeff)[0]
        nmo = self.n_mo
        self.h1 = np.zeros((nmo,nmo))
        self.h2 = np.zeros((nmo,nmo,nmo,nmo))

        self._get_spatialOrb_h1(pyscf_mf,pyscf_mol,cc_info)
        self._get_spatialOrb_h2(pyscf_mf,pyscf_mol,cc_info)

    def _get_spatialOrb_h1(self,pyscf_mf,pyscf_mol,cc_info):
        T = pyscf_mol.intor('int1e_kin_sph')
        V = pyscf_mol.intor('int1e_nuc_sph')
        C = np.asarray(pyscf_mf.mo_coeff)
        C = C[:,:]
        hcore = T + V
        dm = C @ C.T # or pyscf_mf.dm
        j, k = scf.hf.get_jk(pyscf_mol, dm)

        t = hcore + 2*j - k
        self.h1 = reduce(np.dot, (C.conj().T, hcore + 2*j - k, C))
        ecore = np.trace(2*dm @ (hcore + j - .5*k))
        print(" ecore: %12.8f" %ecore)
        print("h1!!!",self.h1)

    def _get_spatialOrb_h2(self,pyscf_mf, pyscf_mol,cc_info):
        C = np.asarray(pyscf_mf.mo_coeff)
        g = pyscf_mol.intor('int2e_sph')
        g = np.einsum("pqrs,pl->lqrs",g,C)
        g = np.einsum("lqrs,qm->lmrs",g,C)
        g = np.einsum("lmrs,rn->lmns",g,C)
        self.h2 = np.einsum("lmns,so->lmno",g,C)



class MeanFieldToJWspin(SpatialOrbInfo):
    def __init__(self, pyscf_mol,pyscf_mf,cc_info,spats):#Set the defaults up for CC calc
        super().__init__(pyscf_mol,pyscf_mf,cc_info)
        self.int_H = cp.deepcopy(self.h1) #None #cp.deepcopy(h)
        self.int_V = cp.deepcopy(self.h2) #None #cp.deepcopy(v)
        self.int_A = np.array(())
        self.int_B = np.array(())
        self.int_C = np.array(())
        self.int_D = np.array(())

        self.n_alpha = self.n_beta = self.n_electrons = self.n_alpha = self.n_beta =None

        self.n_mo = self.n_spin_orbitals =  self.n_mo = self.n_qubits = None
        self.n_spin_orbitals = self.n_particles = self.n_occ_spatial = None

        self.dropcore=cc_info.get('dropcore',0)
        self.cc_calc = cc_info.get('slowSOcalc','CCD') # set the default calc to CCD if not specified
        self.C = self.eps = None
        self.hcore = self.g = self.fock = None

        self.occInfo = self.occSliceInfo = {}
        self.denomInfo = self.integralInfo = {}

        self.E_nuc = pyscf_mf.energy_nuc()
        self.E_scf = pyscf_mf.e_tot  
        print('SCF energy pycc:',self.E_scf,pyscf_mf.e_tot)
        
    def export_FermionOperator(self, shift=0):
        """
        We have spatial orbital integrals, so we need to convert back to spin orbitals
        """
        fermi_op = openfermion.FermionOperator()

        #H
        for p in range(self.int_H.shape[0]):
            pa = 2*p + shift
            pb = 2*p+1 +  shift
            for q in range(self.int_H.shape[1]):
                qa = 2*q +shift
                qb = 2*q+1 +shift
                fermi_op += openfermion.FermionOperator(((pa,1),(qa,0)), self.int_H[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,1),(qb,0)), self.int_H[p,q]) 
        
        #V
        for p in range(self.int_V.shape[0]):
            pa = 2*p +shift
            pb = 2*p+1 +shift
            for q in range(self.int_V.shape[1]):
                qa = 2*q +shift
                qb = 2*q+1 +shift
                for r in range(self.int_V.shape[2]):
                    ra = 2*r +shift
                    rb = 2*r+1 +shift
                    for s in range(self.int_V.shape[3]):
                        sa = 2*s +shift
                        sb = 2*s+1 +shift
                        #aa
                        fermi_op += .5* openfermion.FermionOperator(((pa,1),(qa,1),(sa,0),(ra,0)), self.int_V[p,r,q,s]) 
                        #ab
                        fermi_op += .5*openfermion.FermionOperator(((pa,1),(qb,1),(sb,0),(ra,0)), self.int_V[p,r,q,s]) 
                        #ba
                        fermi_op += .5*openfermion.FermionOperator(((pb,1),(qa,1),(sa,0),(rb,0)), self.int_V[p,r,q,s]) 
                        #bb
                        fermi_op += .5*openfermion.FermionOperator(((pb,1),(qb,1),(sb,0),(rb,0)), self.int_V[p,r,q,s]) 

        #A
        for p in range(self.int_A.shape[0]):
            pa = 2*p
            pb = 2*p+1
            fermi_op += openfermion.FermionOperator(((pa,1)), self.int_A[p]) 
            fermi_op += openfermion.FermionOperator(((pb,1)), self.int_A[p]) 
    
        #B
        for p in range(self.int_B.shape[0]):
            pa = 2*p
            pb = 2*p+1
            fermi_op += openfermion.FermionOperator(((pa,0)), self.int_B[p]) 
            fermi_op += openfermion.FermionOperator(((pb,0)), self.int_B[p]) 
        
        #C
        for p in range(self.int_C.shape[0]):
            pa = 2*p
            pb = 2*p+1
            for q in range(self.int_C.shape[1]):
                qa = 2*q
                qb = 2*q+1
                fermi_op += openfermion.FermionOperator(((pa,1),(qa,1)), self.int_C[p,q]) 
                fermi_op += openfermion.FermionOperator(((pa,1),(qb,1)), self.int_C[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,1),(qa,1)), self.int_C[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,1),(qb,1)), self.int_C[p,q]) 
        
        #D
        for p in range(self.int_D.shape[0]):
            pa = 2*p
            pb = 2*p+1
            for q in range(self.int_D.shape[1]):
                qa = 2*q
                qb = 2*q+1
                fermi_op += openfermion.FermionOperator(((pa,0),(qa,0)), self.int_D[p,q]) 
                fermi_op += openfermion.FermionOperator(((pa,0),(qb,0)), self.int_D[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,0),(qa,0)), self.int_D[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,0),(qb,0)), self.int_D[p,q]) 
        


        return fermi_op


    def spin_block_tei(self,I):
        """
        Function that spin blocks two-electron integrals
        Using np.kron, we project I into the space of the 2x2 identity, tranpose the result
        and project into the space of the 2x2 identity again. This doubles the size of each axis.
        The result is our two electron integral tensor in the spin orbital form.
        """
        identity = np.eye(2)
        I = np.kron(identity, I)
        return np.kron(identity, I.T)

    def spin_block_C_eps(self,pyscf_mf):
        dropcore = self.dropcore
        if 'RHF' in str(type(pyscf_mf)): # running RHF calculation
            Ca = Cb = np.asarray(pyscf_mf.mo_coeff)
            eps_a = eps_b = np.asarray(pyscf_mf.mo_energy)

        elif 'UHF' in str(type(pyscf_mf)): # running UHF calculation
            Ca = np.asarray(pyscf_mf.mo_coeff[0])
            Cb = np.asarray(pyscf_mf.mo_coeff[1])
            eps_a = np.asarray(pyscf_mf.mo_energy[0])
            eps_b = np.asarray(pyscf_mf.mo_energy[1])
            print('eps_a',eps_a)

        C = np.block([
                 [      Ca           ,   np.zeros_like(Cb) ],
                 [np.zeros_like(Ca)  ,          Cb         ]
                ])


        eps = np.append(eps_a, eps_b)

        self._build_hcore(pyscf_mf)

        # Sort the columns of C according to the order of increasing orbital energies
        self.C = C[:, eps.argsort()[dropcore*2:]]
        # Sort orbital energies in increasing order
        self.eps = np.sort(eps)[dropcore*2:]
        self.hcore = self.hcore[:,eps.argsort()[dropcore*2:]]

        # -------------------------
        # 2) Build electronic Hamiltonian
        # -------------------------
        # one-electron integrals in MO basis
    def _build_fock(self):
        print(self.eps)
        self.fock=np.diag(self.eps)

 
    def _build_hcore(self,pyscf_mf):
        C = np.asarray(pyscf_mf.mo_coeff)
        h_core_ao = pyscf_mf.get_hcore()
        h1 = C.T @ h_core_ao @ C
        self.hcore = np.block([
                 [    h1             ,   np.zeros_like(h1) ],
                 [np.zeros_like(h1)  ,          h1         ]
                ])
        


    def _build_tei(self,pyscf_mol):
        # two-electron integrals in MO basis (chemist notation)
        C = self.C
        eri = pyscf_mol.intor("int2e",aosym='s1')
        I = np.asarray(eri)
        I_spinblock = self.spin_block_tei(I)
        # Converts chemist's notation to physicist's notation, and antisymmetrize
        # (pq | rs) ---> <pr | qs>
        # Physicist's notation
        tmp = I_spinblock.transpose(0, 2, 1, 3)
        # Antisymmetrize:
        # <pr||qs> = <pr | qs> - <pr | sq>
        gao = tmp - tmp.transpose(0, 1, 3, 2)

            # Transform gao, which is the spin-blocked 4d array of physicist's notation,
    # antisymmetric two-electron integrals, into the MO basis using MO coefficients
        self.g = np.einsum('pQRS, pP -> PQRS',
              np.einsum('pqRS, qQ -> pQRS',
              np.einsum('pqrS, rR -> pqRS',
              np.einsum('pqrs, sS -> pqrS', gao, C, optimize=True), C, optimize=True), C, optimize=True), C, optimize=True)

    
    def get_integrals(self,pyscf_mol,pyscf_mf):
        #self._build_hcore(pyscf_mf)
        self._build_fock()
        self._build_tei(pyscf_mol)
        self.integralInfo = {"oei":self.fock,"tei":self.g}

    def get_denomsSlow(self):
        import pycc.set_denoms as set_denoms
        virt_aa=self.occSliceInfo["virt_aa"]
        occ_aa=self.occSliceInfo["occ_aa"]
        epsaa=self.eps
        n = np.newaxis
        cc_calc = self.cc_calc
        if "S" in cc_calc: # Get T1 denoms
            self.denomInfo.update({"D1aa":set_denoms.D1denomSlow(epsaa,occ_aa,virt_aa,n)})
        if "D" in cc_calc: # Get T2 denoms
            self.denomInfo.update({"D2aa":set_denoms.D2denomSlow(epsaa,occ_aa,virt_aa,n)})
        if "T" in cc_calc: #Get T3 denoms
            self.denomInfo.update({"D3aa":set_denoms.D3denomSlow(epsaa,occ_aa,virt_aa,n)})
        if "UT2" in cc_calc or "X" in cc_calc or "Qdebug" in cc_calc or "Q" in cc_calc:
            self.denomInfo.update({"D4aa":set_denoms.D4denomSlow(epsaa,occ_aa,virt_aa,n)})

    def get_orb_info(self,pyscf_mol,pyscf_mf,calcType=None):
        dc = self.dropcore*2


        self.n_alpha , self.n_beta = pyscf_mol.nelec #- (dc//2,dc//2)
        self.n_electrons = self.n_alpha + self.n_beta - dc
        self.n_mo = pyscf_mf.mo_coeff.shape[1] - dc
        self.n_spin_orbitals = 2 * self.n_mo - dc
        self.n_qubits = self.n_spin_orbitals - dc
        self.n_particles = (self.n_alpha -dc//2, self.n_beta - dc //2)
        self.n_occ_spatial = self.n_electrons // 2 - self.dropcore
        print(f"n_mo={self.n_mo}, n_spin_orbitals={self.n_spin_orbitals}, n_electrons={self.n_electrons}")

        o = slice(None, self.n_electrons)
        v = slice(self.n_electrons, None)
        self.occSliceInfo={"occ_aa":o,"virt_aa":v}

        self.n_spin_virts = self.n_spin_orbitals - self.n_electrons + dc #overcounted
        self.occInfo={"nocc_aa":self.n_electrons,"nvirt_aa":self.n_spin_virts}

    def collect_data(self, pyscf_mol,pyscf_mf,cc_info):
        self.get_orb_info(pyscf_mol,pyscf_mf,cc_info)
        self.spin_block_C_eps(pyscf_mf)
        self.get_integrals(pyscf_mol,pyscf_mf)
        self.get_denomsSlow()




def generate_mf_data():
    pyscf_mol, pyscf_mf = run_pyscf()
    cc_info = {"dropcore":0}
    obj = MeanFieldData(pyscf_mol,pyscf_mf,cc_info)
    obj.collect_data(pyscf_mol,pyscf_mf,cc_info)
    return obj

pyscf_mol, pyscf_mf = run_pyscf()
cc_info = {"dropcore":0}

spats = SpatialOrbInfo(pyscf_mol,pyscf_mf,cc_info)
obj2 = MeanFieldToJWspin(pyscf_mol,pyscf_mf,cc_info,spats)
obj2.collect_data(pyscf_mol,pyscf_mf,cc_info)
obj2.export_FermionOperator()
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
#H = H_sparse.toarray()                                # dense matrix (only OK for small n_qubits)
H=H_sparse
H_dense = H_sparse.toarray()
print(np.shape(H))
# -------------------------
# 3) Prepare HF reference state (computational basis)
# -------------------------
# Standard ordering: spin-orbital 0..n_qubits-1. Fill first n_electrons orbitals.
hf_occ = [1 if i < obj.n_electrons else 0 for i in range(obj.n_qubits)]
hf_index = sum((bit << i) for i, bit in enumerate(hf_occ))  # little-endian: q0 is LSB
dim = 2**obj.n_qubits
psi0 = np.zeros(dim, dtype=complex)
psi0[hf_index] = 1.0
print("Hartree-Fock bitstring (LSB=q0):", "".join(str(b) for b in reversed(hf_occ)))

# -------------------------
# 4) Build UCCSD generator (singles + doubles)
# -------------------------
# We'll build parameterized FermionOperators for each unique excitation and map them to qubit operators.
# Indexing convention: spin-orbitals indices 0..n_qubits-1. Occupied: 0..(n_occ_spatial*2 -1),
# virtual: the rest.

n_occ_spin = obj.n_electrons 
occ_spin_idxs = list(range(n_occ_spin))
virt_spin_idxs = list(range(n_occ_spin, obj.n_qubits))

singles = []   # each element: (p, q) with p in virt, q in occ -> excitation a_p^\dag a_q
doubles = []   # each element: (p, q, r, s) with p,q in virt; r,s in occ -> a_p^\dag a_q^\dag a_r a_s

# Generate unique spin-orbital excitations (no spin-adaptation here)
for q in occ_spin_idxs:
    for p in virt_spin_idxs:
        singles.append((p, q))

# Doubles: choose two occupied and two virtual (with order to avoid duplicates)
for r_idx in range(len(occ_spin_idxs)):
    for s_idx in range(r_idx + 1, len(occ_spin_idxs)):
        r = occ_spin_idxs[r_idx]
        s = occ_spin_idxs[s_idx]
        for p_idx in range(len(virt_spin_idxs)):
            for q_idx in range(p_idx + 1, len(virt_spin_idxs)):
                p = virt_spin_idxs[p_idx]
                q_ = virt_spin_idxs[q_idx]
                doubles.append((p, q_, r, s))

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
    mat = get_sparse_operator(qop, obj.n_qubits).toarray()
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
    U = np.eye(2**obj.n_qubits, dtype=complex)
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
#molecule = run_pyscf()
## Get OpenFermion Hamiltonian
#hamiltonian = molecule.get_molecular_hamiltonian()
#one_body, two_body = molecule.get_integrals()
#two_body = molecule.antisymtei(two_body)
#
## Convert to a FermionOperator
#hamiltonian_ferm_op = of.get_fermion_operator(hamiltonian)
#
#print(hamiltonian_ferm_op)
#
#
## Map to QubitOperator using the JWT
#hamiltonian_jw = of.jordan_wigner(hamiltonian_ferm_op)
#
## Convert to Scipy sparse matrix
#hamiltonian_jw_sparse = of.get_sparse_operator(hamiltonian_jw)
#
## Compute ground energy
#eigs, _ = linalg.eigsh(hamiltonian_jw_sparse, k=1, which="SA")
#ground_energy = eigs[0]
#
#print("Ground_energy: {}".format(ground_energy))
#print("JWT transformed Hamiltonian:")
#print(hamiltonian_jw)
#H_dense = hamiltonian_jw

v1100 = np.zeros(2**obj2.n_qubits)
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

psi0 = v1100
def energy_from_theta(theta, reps=1):
    """
    Compute energy expectation <psi(theta)| H |psi(theta)> where
    |psi(theta)> = U(theta) |HF>.
    """
    U = trotterized_u_of_theta(theta, reps=reps)
    psi = U.dot(psi0)
    # expectation value
    print("shapes:",np.shape(psi),np.shape(H_dense),np.shape(theta),np.shape(U))
    E = np.vdot(psi, H_dense.dot(psi)).real
    return E

# Quick test: zero parameters should give HF energy (within numerical error)
theta0 = np.zeros(n_params)
E0 = energy_from_theta(theta0, reps=1)  
print(f"Energy at theta=0 (should be HF energy): {E0:.12f}  PySCF RHF energy: {obj.E_scf:.12f},{energy_from_theta(theta0):.12f}")

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
res = minimize(energy_from_theta, x0=theta0, method="BFGS", options=opts, callback=None)
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
    cisolver = fci.FCI(mf)
    e_fci = cisolver.kernel()[0]
    print(f"FCI energy: {e_fci:.12f}")
except Exception as e:
    print("FCI comparison skipped:", e)



if __name__ == '__main__':
    main()


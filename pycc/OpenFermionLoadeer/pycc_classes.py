from pyscf.cc import ccsd
import pyscf
from pyscf import lib
import scipy
import numpy as np
from math import comb
from functools import reduce
import time
import sys
# PySCF
from pyscf import gto, scf, ao2mo

from itertools import combinations
# OpenFermion
from openfermion.ops import InteractionOperator, FermionOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.utils import hermitian_conjugated
from openfermion.linalg import get_sparse_operator  # returns a scipy sparse operator
from openfermion.linalg import (
    get_number_preserving_sparse_operator,
    jw_number_restrict_state,
    jw_hartree_fock_state,
    jw_sz_restrict_state,
)
from openfermionpyscf import run_pyscf
from openfermionpyscf import generate_molecular_hamiltonian
from openfermion.chem import MolecularData
from itertools import product
import openfermion

# SciPy / NumPy for exponentiation and optimization
import scipy.linalg as la
from scipy.optimize import minimize

import copy as cp
from pyscf import gto, scf, mcscf, fci, ao2mo,  cc
from pyscf.cc import ccsd
import pyscf
from pyscf import lib
import pycc

# ---------------------------------------------------------------------
# Outer optimization (Nelder–Mead over lambda)
# ---------------------------------------------------------------------

class MetaVQE:
    def __init__(self, theta0):
        self.theta_opt = np.copy(theta0)

    def outer_objective(self, lam_vec):
        lam = lam_vec[0]  # scalar λ
        E_min, theta_star = run_vqe(lam, self.theta_opt)
        self.theta_opt = theta_star  # warm start next inner VQE
        S2_val = expectation(theta_star, S2)
        print(f"λ = {lam:8.4f}  E = {E_min:8.5f}  <S^2> = {S2_val:6.3f}")
        return E_min



def run_pyscf22():
    mol = gto.Mole()
    mol.atom = [
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 1)],
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



def initialize_pyscf(geometry,basis,multiplicity,charge,mult,ref="RHF"):
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.spin = mult
    mol.charge = charge
    mol.build()

    if ref == "RHF":
        mf = scf.RHF(mol)
    elif ref == "ROHF":
        mf = scf.ROHF(mol)
    else:
        print("Reference not implemented yet. ")
        sys.exit()

    mf.kernel()
    return mol, mf

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
    def __init__(self, pyscf_mol,pyscf_mf,cc_info,spats=None):#Set the defaults up for CC calc
        super().__init__(pyscf_mol,pyscf_mf,cc_info)
        self.int_H = cp.deepcopy(self.h1) #None #cp.deepcopy(h)
        self.int_V = cp.deepcopy(self.h2) #None #cp.deepcopy(v)
        self.int_A = np.array(())
        self.int_B = np.array(())
        self.int_C = np.array(())
        self.int_D = np.array(())
        self.int_Hspin_fermiop = None #np.array(())
        self.int_Vspin_fermiop = None #np.array(())

        self.n_alpha = self.n_beta = self.n_electrons = self.n_alpha = self.n_beta =None

        self.n_mo = self.n_spin_orbitals =  self.n_mo = self.n_qubits = None
        self.n_spin_orbitals = self.n_particles = self.n_occ_spatial = None

        self.dropcore=cc_info.get('dropcore',0)
        self.cc_calc = cc_info.get('slowSOcalc','T') # set the default calc to CCD if not specified
        self.C = self.eps = None
        self.hcore = self.g = self.fock = None

        self.occInfo = self.occSliceInfo = {}
        self.denomInfo = self.integralInfo = {}

        self.E_nuc = pyscf_mf.energy_nuc()
        self.E_scf = pyscf_mf.e_tot  
        print('SCF energy pycc:',self.E_scf,pyscf_mf.e_tot)
        self.psi0 = None
        self.Hdef = None
        self.sparse_basis = None

        self.singles = None
        self.doubles = None
        self.n_params = None
        self.gen_unitary_mats = None
        self.H_ferm = None
        cc_info = {"slowSOcalc":"T"}
        self.pycc_obj = pycc.pycc.SetupCC(pyscf_mf,pyscf_mol,cc_info)
        self.gen_fermion_terms = []
        self.init_theta = None

    def export_FermionOperator(self, shift=0):
        """
        We have spatial orbital integrals, so we need to convert back to spin orbitals
        """
        fermi_op = openfermion.FermionOperator()
        self.int_Hspin_fermiop = openfermion.FermionOperator()
        self.int_Vspin_fermiop = openfermion.FermionOperator()

        #H
        for p in range(self.int_H.shape[0]):
            pa = 2*p + shift
            pb = 2*p+1 +  shift
            for q in range(self.int_H.shape[1]):
                qa = 2*q +shift
                qb = 2*q+1 +shift
                fermi_op += openfermion.FermionOperator(((pa,1),(qa,0)), self.int_H[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,1),(qb,0)), self.int_H[p,q]) 

                self.int_Hspin_fermiop +=openfermion.FermionOperator(((pa,1),(qa,0)), self.int_H[p,q])
                self.int_Hspin_fermiop +=openfermion.FermionOperator(((pb,1),(qb,0)), self.int_H[p,q])
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
                        fermi_op += .5*openfermion.FermionOperator(((pa,1),(qb,1),(sb,0),(ra,0)), self.int_V[p,r,q,s]) 
                        #ba
                        fermi_op += .5*openfermion.FermionOperator(((pb,1),(qa,1),(sa,0),(rb,0)), self.int_V[p,r,q,s]) 
                        #bb
                        fermi_op += .5*openfermion.FermionOperator(((pb,1),(qb,1),(sb,0),(rb,0)), self.int_V[p,r,q,s]) 

                        #aa
                        self.int_Vspin_fermiop += .5* openfermion.FermionOperator(((pa,1),(qa,1),(sa,0),(ra,0)), self.int_V[p,r,q,s])
                        #ab
                        self.int_Vspin_fermiop += .5*openfermion.FermionOperator(((pa,1),(qb,1),(sb,0),(ra,0)), self.int_V[p,r,q,s])
                        #ba
                        self.int_Vspin_fermiop += .5*openfermion.FermionOperator(((pb,1),(qa,1),(sa,0),(rb,0)), self.int_V[p,r,q,s])
                        #bb
                        self.int_Vspin_fermiop += .5*openfermion.FermionOperator(((pb,1),(qb,1),(sb,0),(rb,0)), self.int_V[p,r,q,s])


        self.int_Hspin_fermiop = get_sparse_operator(self.int_Hspin_fermiop,self.n_spin_orbitals)
        self.int_Vspin_fermiop = get_sparse_operator(self.int_Vspin_fermiop,self.n_spin_orbitals)

        return fermi_op


    def build_hf_state(self):
        n_spin_orb = self.n_spin_orbitals
        hf_occ = [1] * self.n_electrons + [0] * (n_spin_orb - self.n_electrons)
        hf_index = int("".join(str(b) for b in hf_occ), 2)
        print('hf_index:',hf_index)
        #the following two lines are construct the HF determinant in the full 2^n x 2^n Fock space. need to simplify this
        self.psi0 = np.zeros(2 ** n_spin_orb, dtype=complex)
        self.psi0[hf_index] = 1.0
        #state = self.psi0
        #restricted_state = jw_number_restrict_state(state, self.n_electrons, n_qubits=n_spin_orb)

        # This logic restricts particle number of the state vector for conservation.
        # For some reason, the output is |0000...1> when openfermion uses the opposite convention, so i have to resort
        # ** BUT ** this is only for particle number conservation. We also need to implement Sz conservation of the state.
        #self.psi0 = jw_number_restrict_state(jw_hartree_fock_state(self.n_electrons,n_spin_orb), self.n_electrons,n_spin_orb)
        #self.psi0 = self.psi0[::-1]
        #print('restrict particle num:',self.psi0.shape)

        # This logic restricts ** BOTH ** particle number and Sz quantum numbers on the state
        sz_target = (self.n_alpha - self.n_beta) / 2
        self.psi0 = jw_sz_restrict_state(self.psi0,sz_target,self.n_electrons,n_spin_orb)
        print('restrict sz:',self.psi0.shape)



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
                 [      Ca           ,       np.zeros_like(Cb) ],
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
        # Converts c    mol = gto.Mole()hemist's notation to physicist's notation, and antisymmetrize
        # (pq | rs) ---> <pr | qs>
        # Physicist's notation
        tmp = I_spinblock.transpose(0, 2, 1, 3)
        # Anti-symmeterize:
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
            self.denomInfo.update({"D2aa":set_denoms.D2denomSlow(epsaa,occ_aa,virt_aa,n)})
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
        self.doubles_ordering = None

    def get_molecular_H(self,pyscf_mol):
        print("geo:",pyscf_mol.atom)
        geometry = pyscf_mol.atom
        basis = pyscf_mol.basis
        multiplicity = pyscf_mol.multiplicity  
        charge = pyscf_mol.charge
        # The below is in the full 2^n x 2^n Fock space, so we need to reduce the dimensionality to just the
        # Hilbert space in question...
        #self.Hdef = get_sparse_operator(generate_molecular_hamiltonian(geometry,basis,multiplicity,charge))
        
        # So I start with a Fermionic operator
        H_ferm = get_fermion_operator(generate_molecular_hamiltonian(geometry,basis,multiplicity,charge))

        self.H_ferm = H_ferm
#       **** THIS SECTION WAS SUPPOSED TO INITIALIZE THETA WITH CCSD AMPS, BUT DOESNT APPEAR TO WORK VERY WELL
#        pyscf_obj = run_pyscf(MolecularData(geometry, basis, multiplicity, charge),run_ccsd=True)
#        H_ferm = get_fermion_operator(pyscf_obj.get_molecular_hamiltonian())
#
#        ccsd_t1 = pyscf_obj.ccsd_single_amps
#        ccsd_t2 = pyscf_obj.ccsd_double_amps
#        self.init_theta = np.concatenate(( ccsd_t2.flatten(),ccsd_t1.flatten()))

        # This should return (H_sparse, basis) in your version
        H_sparse = get_number_preserving_sparse_operator(
            H_ferm,
            self.n_spin_orbitals,
            self.n_electrons,
            spin_preserving=True
        )

        self.Hdef = H_sparse
        print("Original Hamiltonian shape:",get_sparse_operator(generate_molecular_hamiltonian(geometry,basis,multiplicity,charge)).shape)
        print("Restricted Hamiltonian shape:", H_sparse.shape)
        #print("Number of determinants (basis size):", len(self.sparse_basis))


    def generate_singles(self,occ_spin_idxs,virt_spin_idxs):
        occ = list(range(self.n_electrons))
        virt = list(range(self.n_electrons, self.n_qubits))
        alpha=[]
        beta=[]
        total=[]
        for i in occ:
            for a in virt:
                if (a % 2) != (i % 2):
                    # spin mismatch => would flip spin, skip
                    continue
                #print("a^ i",a,i)
                total.append((a,i))
    
        alpha = [d for d in total if d[0] % 2 == 0]  # i is alpha
        beta = [d for d in total if d[0] % 2 == 1]
        # interleave them
        singles_alt = [x for pair in zip(alpha, beta) for x in pair]
        #print("singles alt:",singles_alt)

        return singles_alt
    
    def generate_doubles(self,occ_spin_idxs,virt_spin_idxs):
        # --- Doubles: use unique pairs (i<j, a<b) and enforce S_z conservation ---
        # We choose i_idx < j_idx and a_idx < b_idx to avoid duplicates.
        doubles = []
        nocc = int(self.n_electrons/2)
        nvirt = int(self.n_mo - nocc)
        #print("nocc, nvirt:",nocc,nvirt)
        # This handles the mixed spin abab amps; must also do pure alpha/beta cases as well
        for A in range(nvirt):
            a=2*nocc+2*A
            for B in range(nvirt):
                b=2*nocc+2*B+1
                for J in range(nocc):
                    j=2*J
                    for I in range(nocc):
                        i=2*I+1
                        if (i % 2) != (b % 2):
                            continue
                        if (j % 2) != (a % 2):
                            continue
                        #print("a^b^ji:",a,b,j,i)
                        doubles.append((a,b,j,i))

        # Now, add the pure spin amps to the list
        occ = list(range(self.n_electrons))
        virt = list(range(self.n_electrons, self.n_qubits))
        same_spin_doubles = [(a, b, i,j)
                             for (i, j) in combinations(occ, 2)
                             for (a, b) in combinations(virt, 2)
                             if (i % 2) == (j % 2) == (a % 2) == (b % 2)]
    
        alpha = [d for d in same_spin_doubles if d[0] % 2 == 0]  # i is alpha
        beta  = [d for d in same_spin_doubles if d[0] % 2 == 1]  # i is beta
    
        # interleave them
        doubles_alt = [x for pair in zip(alpha, beta) for x in pair]
    
        #print("doubles_alt:",doubles_alt)
        doubles.extend(doubles_alt)
        self.doubles_ordering = doubles

        return doubles



    # Helper to build FermionOperator for a single excitation E_pq = a_p^\dag a_q
    def single_excitation_op(self,p, q):
        # returns FermionOperator representing (a_p^\dag a_q)
        return FermionOperator(((p, 1), (q, 0)), 1.0)
    
    # Helper for double excitation E_pqrs = a_p^\dag a_q^\dag a_r a_s
    def double_excitation_op(self,p, q, r, s, coeff = 1.0):
        return FermionOperator(((p, 1), (q, 1), (r, 0), (s, 0)), coeff)




    def build_params(self):
        n_occ_spin = self.n_electrons
        occ_spin_idxs = list(range(n_occ_spin))
        virt_spin_idxs = list(range(n_occ_spin, self.n_qubits))

        self.singles = self.generate_singles(occ_spin_idxs,virt_spin_idxs)
        self.doubles = self.generate_doubles(occ_spin_idxs,virt_spin_idxs)
        # Build list of FermionOperators for generator terms
        gen_fermion_terms = []
        # doubles
        for (a, b, j, i) in self.doubles:
            gen_fermion_terms.append(self.double_excitation_op(a, b, j, i))
        
        for (a, i) in self.singles:
            gen_fermion_terms.append(self.single_excitation_op(a, i))

        self.gen_fermion_terms = gen_fermion_terms

        # Convert each fermion term to qubit operator (JW)
        gen_qubit_terms = [jordan_wigner(term) for term in gen_fermion_terms]
        
        # Convert to dense matrices (qubit space)
        self.gen_unitary_mats = []
        for qop in gen_fermion_terms: #gen_qubit_terms:
            # Ensure hermiticity handling: generator element is (T_i - T_i^\dag)
            # We'll store the qubit operator for T_i (not yet anti-Hermitian)
            #mat = get_sparse_operator(qop, self.n_qubits).toarray()
            mat = get_number_preserving_sparse_operator(qop,
                    self.n_spin_orbitals,
                    self.n_electrons,
                    spin_preserving=True).toarray()
            self.gen_unitary_mats.append(mat)

        self.n_params = len(self.singles) + len(self.doubles)


    def prepare_state_ops(self):
        n_occ_spin = self.n_electrons
        occ_spin_idxs = list(range(n_occ_spin))
        virt_spin_idxs = list(range(n_occ_spin, self.n_qubits))

        self.singles = self.generate_singles(occ_spin_idxs,virt_spin_idxs)
        self.doubles = self.generate_doubles(occ_spin_idxs,virt_spin_idxs)

        # COMMENTED OUT FOR REDUNDANCY ZWW 10/10/25
        # Build list of FermionOperators for generator terms
        #for (p, q) in self.singles:
        #    self.gen_fermion_terms.append(self.single_excitation_op(p, q))
        # doubles
        #for (p, q, r, s) in self.doubles:
        #    self.gen_fermion_terms.append(self.double_excitation_op(p, q, r, s))

    def prepare_state(self,theta,reps=1):
        self.prepare_state_ops()
        new_ref = self.psi0
        #reversed
        for k, qop in zip(range(0, len(theta)),self.gen_fermion_terms):
            mat = get_number_preserving_sparse_operator(qop,
                    self.n_spin_orbitals,
                    self.n_electrons,
                    spin_preserving=True).toarray()
            antiH = mat - mat.conj().T
            new_ref = scipy.sparse.linalg.expm_multiply((theta[k]*antiH), new_ref)

        return new_ref

    # -------------------------
    # 5) Build trotterized U(theta) action
    # -------------------------
    def trotterized_u_of_theta(self,theta, reps=1):
        """
        Build dense unitary matrix U(theta) = Prod_{rep} Prod_i exp( (theta_i/ reps) * (G_i - G_i^\dag) )
        where G_i are fermionic excitation qubit-mapped matrices (here gen_unitary_mats[i]).
        Note: ordering is fixed as the generation order above.
        Returns dense matrix (2^n x 2^n).
        """
        U = np.eye(self.Hdef.shape[0])#np.eye(2**self.n_qubits, dtype=complex)
        # For each T_i construct A_i = G_i (matrix), then anti-hermitian generator = (A_i - A_i^\dag)
        for _rep in range(reps):
            for i, A in enumerate(self.gen_unitary_mats):
                theta_i = theta[i]
                G = A
                antiH = G - G.conj().T
                # exponentiate: exp( (theta_i / reps) * antiH )
                mat = la.expm((theta_i / reps) * antiH)
                U = mat.dot(U)   # left-multiply: apply this small-unitary next
        return U



    def compute_expectation_value(self,operator,theta,reps=1):
        psi = self.prepare_state(theta,reps=1)
        return np.vdot(psi, operator.dot(psi)).real


    def energy_from_theta(self, theta, reps=1):
        """
        Compute energy expectation <psi(theta)| H |psi(theta)> where
        |psi(theta)> = U(theta) |HF>.
        """
        H_dense = self.Hdef
        #U = self.trotterized_u_of_theta(theta, reps=reps)
        #psi = U.dot(self.psi0)
        psi = self.prepare_state(theta,reps=1)
        #[T] correction
    #    thetaT = transform_theta(theta)
        ## compute [T] fourth order diagrams-> sqrBrakT_scale
        # expectation value
        E = np.vdot(psi, H_dense.dot(psi)).real  #+ self.uccsd_FO_triples_corrections(theta)
        return E

    def extract_current_T2(self,theta):
        len_T2 = len(self.doubles)
        nv = self.occInfo["nvirt_aa"]
        no = self.occInfo["nocc_aa"]
        T2 = np.zeros((nv,nv,no,no))
        #for op_order, amp in zip(self.doubles_ordering, theta[len_T1:len_T1+len_T2]):
        for op_order, amp in zip(self.doubles_ordering, theta[:len_T2]):
            a=op_order[0]-no
            b=op_order[1]-no
            j=op_order[2]
            i=op_order[3]
            #print("a,b,i,j",a,b,j,i)
            T2[a,b,j,i]=amp
            T2[b,a,j,i]=-1.0*amp
            T2[a,b,i,j]=-1.0*amp
            T2[b,a,i,j]=amp

        T2 = T2#*0.25
        return T2

    def get_modified_VQE_op(self,theta,gamma=1.0,reps=1):
        T2 = self.extract_current_T2(theta)
        #T2=T2.transpose(2,3,0,1)
        import pycc.OpenFermionLoadeer.pt_helpers as pt_helper
        T2eff = pt_helper.build_penalty_op(
                        self.fock, self.g, T2.transpose(2,3,0,1), 
                        self.occSliceInfo["occ_aa"], self.occSliceInfo["virt_aa"],
                        self.denomInfo["D2aa"],self.denomInfo["D3aa"]
                )
        T2eff = T2eff.transpose(2,3,0,1)
        nv, no = range(self.occInfo["nvirt_aa"]) , range(self.occInfo["nocc_aa"])
        nocc=self.occInfo["nvirt_aa"]
        nvirt=self.occInfo["nocc_aa"]
        A = FermionOperator()

        for a, b, i, j in product(nv,nv,no,no):
            t2amp = T2eff[a,b,j,i]
            #print(a,b,i,j,nocc,nvirt)
            term = FermionOperator(((a+nocc,1),(b+nocc,1),(j,0),(i,0)),t2amp)
            A += term

        #A = FermionOperator('', 0.0)
        #for a, b, i, j in product(nv, nv, no, no):
        #    A += self.double_excitation_op(a, b, j, i, T2eff[a, b, j, i])
        from openfermion.utils import hermitian_conjugated
        P_fermion = A * hermitian_conjugated(A) 
#        for term, coeff in A.terms.items():
#            print(term, coeff)

        #P_qubit = jordan_wigner(P_fermion) #.compress()
        return gamma *P_fermion#gamma * get_sparse_operator(P_fermion)#P_qubit) 

    def modified_cost_function(self,theta,reps=1):
        #print("type",type(self.H_ferm),type(self.get_modified_VQE_op(theta)))
        #print(self.H_ferm.shape, self.get_modified_VQE_op(theta))
        H_mods = self.H_ferm + self.get_modified_VQE_op(theta,10)
        H_sparse = get_number_preserving_sparse_operator(
            H_mods,
            self.n_spin_orbitals,
            self.n_electrons,
            spin_preserving=True
        )
        E0 = self.compute_expectation_value(H_sparse,theta)
        #print('total cost:',E0)
        return E0

    def cost_function(self,theta,reps=1):
        #H_dense = self.Hdef
        #psi = self.prepare_state(theta,reps=1)
        E0 = self.energy_from_theta(theta, reps=1)

        #separate lambda
        lamb0 = 0.0001  # for now just set this as a constant #theta[-1]
        lamb1 = 0.1
        T2=self.extract_current_T2(theta)
        T2=T2.transpose(2,3,0,1)
        import pycc.OpenFermionLoadeer.pt_helpers
        t3_penalty, variance = pycc.OpenFermionLoadeer.pt_helpers.uccsd_FO_triples_corrections(F,W,T2,o,v,D3)
        #print("t3_penalty:",t3_penalty,t3_penalty**2)
        print("total cost:",E0 + lamb0*t3_penalty)
        return E0 + (1- lamb0*t3_penalty) #+ lamb1*variance

    def callback(self,xk):
        e = self.energy_from_theta(xk, reps=1)
        print(f"callback: energy={e:.12f}")

    def drive_vqe(self):
        # get openfermion molecular Hamiltonian and HF initial state
#        self.get_molecular_H(pyscf_mol)
#        self.build_hf_state()
        self.build_params()


        # Quick test: zero parameters should give HF energy (within numerical error)
        theta0 = np.zeros(self.n_params)
        #theta0 = self.init_theta
        E0 = self.energy_from_theta(theta0, reps=1)
        print(f"Energy at theta=0 (should be HF energy): {E0:.12f}  PySCF RHF energy: {self.E_scf:.12f},{self.energy_from_theta(theta0):.12f}")
        opts = {"maxiter": 500, "disp": True, "gtol":10E-5}
        print("Starting optimization... (this may take some time for larger ansatz sizes)")
        t_start = time.time()
        #res = minimize(self.energy_from_theta, x0=theta0, method="BFGS", options=opts, callback=self.callback)
        #res = minimize(self.cost_function, x0=theta0, method="BFGS", options=opts, callback=self.callback)
        res = minimize(self.modified_cost_function, x0=theta0, method="CG", options=opts, callback=self.callback)
        t_end = time.time()
        print("Optimization finished in %.2f s" % (t_end - t_start))
        print("Success:", res.success)
        print("Final energy:", res.fun)
        len_T1 = len(self.singles)
        len_T2 = len(self.doubles)
        self.print_final_amps(res,len_T1,len_T2)
        # **NEED TO EVALUATE ORIGINAL, UNBIASED <H> fot energy
        final_E = self.energy_from_theta(res.x, reps=1)
        print("Final evaluation of (unbiased) <H>:",final_E)


    def print_final_amps(self,res,len_T1,len_T2):
        for op, t2amp in zip(self.doubles,res.x[:len_T2]):
            print(f"{op!s:<12} | {t2amp:.16f}")

        for op, t1amp in zip(self.singles,res.x[len_T2:len_T1+len_T2]):
            print(f"{op!s:<14} | {t1amp:.16f}")


        print(self.init_theta)
    def collect_data(self, pyscf_mol,pyscf_mf,cc_info):
        self.get_orb_info(pyscf_mol,pyscf_mf,cc_info)
        self.spin_block_C_eps(pyscf_mf)
        self.get_integrals(pyscf_mol,pyscf_mf)
        self.get_denomsSlow()

        # get openfermion molecular Hamiltonian and HF initial state
        self.get_molecular_H(pyscf_mol)
        self.build_hf_state()



def generate_mf_data():
    pyscf_mol, pyscf_mf = run_pyscf22()
    cc_info = {"dropcore":0}
    obj = MeanFieldData(pyscf_mol,pyscf_mf,cc_info)
    obj.collect_data(pyscf_mol,pyscf_mf,cc_info)
    return obj

#pyscf_mol, pyscf_mf = run_pyscf()
#cc_info = {"dropcore":0}
#
#spats = SpatialOrbInfo(pyscf_mol,pyscf_mf,cc_info)
#obj2 = MeanFieldToJWspin(pyscf_mol,pyscf_mf,cc_info,spats)
#obj2.collect_data(pyscf_mol,pyscf_mf,cc_info)
#obj2.export_FermionOperator()
#obj = obj2
#
#print(type(obj.g),np.shape(obj.g),np.shape(obj.hcore),obj.E_nuc)
#interaction = InteractionOperator(constant=obj.E_nuc, one_body_tensor=obj.int_H, two_body_tensor=0.5*obj.int_V)
## FermionOperator and ensure Hermitian
#fermion_ham = get_fermion_operator(interaction)

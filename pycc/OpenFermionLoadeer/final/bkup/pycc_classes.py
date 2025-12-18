from pyscf import gto, scf, mcscf, fci, ao2mo,  cc
from pyscf.cc import ccsd
import pyscf
from pyscf import lib

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
def run_pyscf():
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
        self.int_Hspin_fermiop = None #np.array(())
        self.int_Vspin_fermiop = None #np.array(())

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
                        #ab
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
        psi = np.zeros(2 ** n_spin_orb, dtype=complex)
        psi[hf_index] = 1.0
        return psi


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

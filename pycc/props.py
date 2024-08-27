import numpy as np
from pyscf.data import nist

def dipole_moment(pyscf_mol,pyscf_mf,rdm1):
    mo_coeff = pyscf_mf.mo_coeff
    with pyscf_mol.with_common_orig((0,0,0)):
        ao_dip = pyscf_mol.intor_symmetric('int1e_r', comp=3)


    scf_dm=pyscf_mf.make_rdm1()
    scf_dip = np.einsum('xij,ji->x', ao_dip, scf_dm).real

    ao_dip_x = ao_dip[0]
    ao_dip_y = ao_dip[1]
    ao_dip_z = ao_dip[2]

    mo_dip_x= (mo_coeff.T@ao_dip_x)@mo_coeff
    mo_dip_y= (mo_coeff.T@ao_dip_y)@mo_coeff
    mo_dip_z= (mo_coeff.T@ao_dip_z)@mo_coeff

    corr_dip_x=np.einsum('ij,ji->', mo_dip_x, rdm1).real
    corr_dip_y=np.einsum('ij,ji->', mo_dip_y, rdm1).real
    corr_dip_z=np.einsum('ij,ji->', mo_dip_z, rdm1).real

    corr_dip=np.asarray([corr_dip_x,corr_dip_y,corr_dip_z])
    #sys.exit()
    charges = pyscf_mol.atom_charges()
    coords  = pyscf_mol.atom_coords()
    nucl_dip = np.einsum('i,ix->x', charges, coords)
    print(type(nucl_dip),np.shape(nucl_dip))
    mol_dip = nucl_dip - scf_dip
    mol_dip *= nist.AU2DEBYE
    print('SCF Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)


    print('corr dip:',corr_dip)
    corr_mol_dip=nucl_dip - corr_dip
    corr_mol_dip *= nist.AU2DEBYE
    print('Correlated Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *corr_mol_dip)


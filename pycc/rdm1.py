import numpy as np

# pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
# Copyright (C) 2020 A. Eugene DePrince III
#
# This file is part of the pdaggerq package.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
spin-orbital pccd amplitude equations
"""
import numpy as np
from numpy import einsum
import pandas as pd

def build_Spinrdm1(driveCCobj):
    T2=driveCCobj.tamps["t2aa"]
    no=np.shape(T2)[0]
    nv=np.shape(T2)[2]
    T1=driveCCobj.tamps.get("t1aa",np.zeros((no,nv)))

    T1dag = T1.transpose(1,0)
    T2dag = T2.transpose(2,3,0,1)

    nmo = driveCCobj.nmo
    o = driveCCobj.occSliceInfo["occ_aa"]
    v = driveCCobj.occSliceInfo["virt_aa"]

    return ccsd_d1(T1dag,T2dag,T1,T2,np.eye(2*nmo),o,v)

def spin_to_spatial_rdm1(self,spin_rdm):
    spin_dim=np.shape(spin_rdm)[0]
    print('spin_dim',spin_dim)
    spatial_dim=int(spin_dim/2)
    print('spatial dim:',spatial_dim)
    rdm_alphaSpatial=np.zeros((spatial_dim,spatial_dim))
    rdm_betaSpatial=np.zeros((spatial_dim,spatial_dim))
    # get oo portion
    count_i=0
    for i in range(0,spin_dim,2):
        count_j=0
        for j in range(0,spin_dim,2):
            #print(i,j,count_i,count_j)
            rdm_alphaSpatial[count_i,count_j]=spin_rdm[i,j]
            count_j+=1
        count_i+=1

    # store alpha/beta portion of contraction on alpha DM
    count_i=0
    for i in range(0,spin_dim,2):
        count_j=0
        for j in range(1,spin_dim,2):
            if np.abs(spin_rdm[i,j])>10**-15:
                print(i,j,int(i/2),int(j/2))
                rdm_alphaSpatial[int(i/2),int(j/2)]=spin_rdm[i,j]
                rdm_alphaSpatial[int(j/2),int(i/2)]=spin_rdm[j,i]
            count_j+=1
        count_i+=1



#    print(rdm_alphaSpatial[0,6],rdm_alphaSpatial[6,0])
#    sys.exit()
    # get beta portion
    count_i=0
    for i in range(1,spin_dim,2):
        count_j=0
        for j in range(1,spin_dim,2):
            #print(i,j,count_i,count_j)
            rdm_betaSpatial[count_i,count_j]=spin_rdm[i,j]
            count_j+=1
        count_i+=1

    # store alpha/beta portion of contraction on beta DM
    for i in range(1,spin_dim,2):
        for j in range(0,spin_dim,2):
            if np.abs(spin_rdm[i,j])>10**-15:
                print(i,j,int(i/2),int(j/2))
                rdm_betaSpatial[int(i/2),int(j/2)]=spin_rdm[i,j]
                rdm_betaSpatial[int(j/2),int(i/2)]=spin_rdm[j,i]


    opdm_alpha=pd.DataFrame(rdm_alphaSpatial[:7,:7])
    print('alpha spatial orb 1RDM:\n',opdm_alpha.round(8))


    opdm_beta=pd.DataFrame(rdm_betaSpatial[:7,:7])
    print('beta spatial orb 1RDM:\n',opdm_beta.round(8))


    print(np.shape(opdm_alpha))

    new=0.0
    for i in range(spatial_dim):
        for j in range(spatial_dim):
            new+= abs(rdm_alphaSpatial[i,j] - rdm_betaSpatial[i,j])

    print('diff b/t alpha and beta: ', new)
    return rdm_alphaSpatial, rdm_betaSpatial #opdm_alpha,opdm_beta

    

def ccsd_d1(t1, t2, l1, l2, kd, o, v):
    """
    Compute CCSD 1-RDM

    :param t1: spin-orbital t1 amplitudes (nvirt x nocc)
    :param t2: spin-orbital t2 amplitudes (nvirt x nvirt x nocc x nocc)
    :param l1: lagrange multiplier for singles (nocc x nvirt)
    :param l2: lagrange multiplier for doubles (nocc x nocc x nvirt x nvirt)
    :param kd: identity matrix (|spin-orb| x |spin-orb|)
    :param o: slice(None, occ) where occ is number of occupied spin-orbitals
    :param v: slice(occ, None) whwere occ is number of occupied spin-orbitals
    """
    opdm = np.zeros_like(kd)

    #    D1(m,n):
    # 	  1.0000 d(m,n)
    # 	 ['+1.000000', 'd(m,n)']
    opdm[o, o] += 1.0 * einsum('mn->mn', kd[o, o])

    # 	 -1.0000 l1(n,a)*t1(a,m)
    # 	 ['-1.000000', 'l1(n,a)', 't1(a,m)']
    opdm[o, o] += -1.0 * einsum('na,am->mn', l1, t1)

    # 	 -0.5000 l2(i,n,b,a)*t2(b,a,i,m)
    # 	 ['-0.500000', 'l2(i,n,b,a)', 't2(b,a,i,m)']
    opdm[o, o] += -0.5 * einsum('inba,baim->mn', l2, t2)

    #    D1(e,f):

    #	  1.0000 l1(i,e)*t1(f,i)
    #	 ['+1.000000', 'l1(i,e)', 't1(f,i)']
    opdm[v, v] += 1.0 * einsum('ie,fi->ef', l1, t1)

    #	  0.5000 l2(i,j,e,a)*t2(f,a,i,j)
    #	 ['+0.500000', 'l2(i,j,e,a)', 't2(f,a,i,j)']
    opdm[v, v] += 0.5 * einsum('ijea,faij->ef', l2, t2)

    #    D1(e,m):

    #	  1.0000 l1(m,e)
    #	 ['+1.000000', 'l1(m,e)']
    opdm[v, o] += 1.0 * einsum('me->em', l1)

    #    D1(m,e):

    #	  1.0000 t1(e,m)
    #	 ['+1.000000', 't1(e,m)']
    opdm[o, v] += 1.0 * einsum('em->me', t1)

    #	 -1.0000 l1(i,a)*t2(e,a,i,m)
    #	 ['-1.000000', 'l1(i,a)', 't2(e,a,i,m)']
    opdm[o, v] += -1.0 * einsum('ia,eaim->me', l1, t2)

    #	 -1.0000 l1(i,a)*t1(e,i)*t1(a,m)
    #	 ['-1.000000', 'l1(i,a)', 't1(e,i)', 't1(a,m)']
    opdm[o, v] += -1.0 * einsum('ia,ei,am->me', l1, t1, t1,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 l2(i,j,b,a)*t1(e,j)*t2(b,a,i,m)
    #	 ['-0.500000', 'l2(i,j,b,a)', 't1(e,j)', 't2(b,a,i,m)']
    opdm[o, v] += -0.5 * einsum('ijba,ej,baim->me', l2, t1, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 l2(i,j,b,a)*t1(b,m)*t2(e,a,i,j)
    #	 ['-0.500000', 'l2(i,j,b,a)', 't1(b,m)', 't2(e,a,i,j)']
    opdm[o, v] += -0.5 * einsum('ijba,bm,eaij->me', l2, t1, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    return opdm


def ccsd_d2(t1, t2, l1, l2, kd, o, v):
    """
    Compute CCSD 2-RDM

    :param t1: spin-orbital t1 amplitudes (nvirt x nocc)
    :param t2: spin-orbital t2 amplitudes (nvirt x nvirt x nocc x nocc)
    :param l1: lagrange multiplier for singles (nocc x nvirt)
    :param l2: lagrange multiplier for doubles (nocc x nocc x nvirt x nvirt)
    :param kd: identity matrix (|spin-orb| x |spin-orb|)
    :param o: slice(None, occ) where occ is number of occupied spin-orbitals
    :param v: slice(occ, None) whwere occ is number of occupied spin-orbitals
    """
    nso = kd.shape[0]
    tpdm = np.zeros((nso, nso, nso, nso))
    #    D2(i,j,k,l):

    #	 ['+1.000000', 'd(j,l)', 'd(i,k)']
    #	  1.0000 d(j,l)*d(i,k)
    tpdm[o, o, o, o] += 1.0 * einsum('jl,ik->ijkl', kd[o, o], kd[o, o])

    #	 ['-1.000000', 'd(i,l)', 'd(j,k)']
    #	 -1.0000 d(i,l)*d(j,k)
    tpdm[o, o, o, o] += -1.0 * einsum('il,jk->ijkl', kd[o, o], kd[o, o])

    #	 ['-1.000000', 'd(j,l)', 'l1(k,a)', 't1(a,i)']
    #	 -1.0000 d(j,l)*l1(k,a)*t1(a,i)
    tpdm[o, o, o, o] += -1.0 * einsum('jl,ka,ai->ijkl', kd[o, o], l1, t1,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	 ['+1.000000', 'd(i,l)', 'l1(k,a)', 't1(a,j)']
    #	  1.0000 d(i,l)*l1(k,a)*t1(a,j)
    tpdm[o, o, o, o] += 1.0 * einsum('il,ka,aj->ijkl', kd[o, o], l1, t1,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 ['+1.000000', 'd(j,k)', 'l1(l,a)', 't1(a,i)']
    #	  1.0000 d(j,k)*l1(l,a)*t1(a,i)
    tpdm[o, o, o, o] += 1.0 * einsum('jk,la,ai->ijkl', kd[o, o], l1, t1,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 ['-1.000000', 'd(i,k)', 'l1(l,a)', 't1(a,j)']
    #	 -1.0000 d(i,k)*l1(l,a)*t1(a,j)
    tpdm[o, o, o, o] += -1.0 * einsum('ik,la,aj->ijkl', kd[o, o], l1, t1,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	 ['-0.500000', 'd(j,l)', 'l2(m,k,b,a)', 't2(b,a,m,i)']
    #	 -0.5000 d(j,l)*l2(m,k,b,a)*t2(b,a,m,i)
    tpdm[o, o, o, o] += -0.5 * einsum('jl,mkba,bami->ijkl', kd[o, o], l2, t2,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	 ['+0.500000', 'd(i,l)', 'l2(m,k,b,a)', 't2(b,a,m,j)']
    #	  0.5000 d(i,l)*l2(m,k,b,a)*t2(b,a,m,j)
    tpdm[o, o, o, o] += 0.5 * einsum('il,mkba,bamj->ijkl', kd[o, o], l2, t2,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 ['+0.500000', 'd(j,k)', 'l2(m,l,b,a)', 't2(b,a,m,i)']
    #	  0.5000 d(j,k)*l2(m,l,b,a)*t2(b,a,m,i)
    tpdm[o, o, o, o] += 0.5 * einsum('jk,mlba,bami->ijkl', kd[o, o], l2, t2,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 ['-0.500000', 'd(i,k)', 'l2(m,l,b,a)', 't2(b,a,m,j)']
    #	 -0.5000 d(i,k)*l2(m,l,b,a)*t2(b,a,m,j)
    tpdm[o, o, o, o] += -0.5 * einsum('ik,mlba,bamj->ijkl', kd[o, o], l2, t2,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	 ['+0.500000', 'l2(k,l,b,a)', 't2(b,a,i,j)']
    #	  0.5000 l2(k,l,b,a)*t2(b,a,i,j)
    tpdm[o, o, o, o] += 0.5 * einsum('klba,baij->ijkl', l2, t2)

    #	 ['-1.000000', 'l2(k,l,b,a)', 't1(b,j)', 't1(a,i)']
    #	 -1.0000 l2(k,l,b,a)*t1(b,j)*t1(a,i)
    tpdm[o, o, o, o] += -1.0 * einsum('klba,bj,ai->ijkl', l2, t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #    D2(i,j,k,a):

    #	 -1.0000 d(j,k)*t1(a,i)
    tpdm[o, o, o, v] += -1.0 * einsum('jk,ai->ijka', kd[o, o], t1)

    #	  1.0000 d(i,k)*t1(a,j)
    tpdm[o, o, o, v] += 1.0 * einsum('ik,aj->ijka', kd[o, o], t1)

    #	  1.0000 d(j,k)*l1(l,b)*t2(a,b,l,i)
    tpdm[o, o, o, v] += 1.0 * einsum('jk,lb,abli->ijka', kd[o, o], l1, t2,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 d(i,k)*l1(l,b)*t2(a,b,l,j)
    tpdm[o, o, o, v] += -1.0 * einsum('ik,lb,ablj->ijka', kd[o, o], l1, t2,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 l1(k,b)*t2(a,b,i,j)
    tpdm[o, o, o, v] += 1.0 * einsum('kb,abij->ijka', l1, t2)

    #	  1.0000 d(j,k)*l1(l,b)*t1(a,l)*t1(b,i)
    tpdm[o, o, o, v] += 1.0 * einsum('jk,lb,al,bi->ijka', kd[o, o], l1, t1, t1,
                                     optimize=['einsum_path', (1, 2), (1, 2),
                                               (0, 1)])

    #	 -1.0000 d(i,k)*l1(l,b)*t1(a,l)*t1(b,j)
    tpdm[o, o, o, v] += -1.0 * einsum('ik,lb,al,bj->ijka', kd[o, o], l1, t1, t1,
                                      optimize=['einsum_path', (1, 2), (1, 2),
                                                (0, 1)])

    #	 -1.0000 P(i,j)l1(k,b)*t1(a,j)*t1(b,i)
    contracted_intermediate = -1.0 * einsum('kb,aj,bi->ijka', l1, t1, t1,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    tpdm[o, o, o, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijka->jika', contracted_intermediate)

    #	  0.5000 d(j,k)*l2(l,m,c,b)*t1(a,m)*t2(c,b,l,i)
    tpdm[o, o, o, v] += 0.5 * einsum('jk,lmcb,am,cbli->ijka', kd[o, o], l2, t1,
                                     t2,
                                     optimize=['einsum_path', (1, 3), (1, 2),
                                               (0, 1)])

    #	  0.5000 d(j,k)*l2(l,m,c,b)*t1(c,i)*t2(a,b,l,m)
    tpdm[o, o, o, v] += 0.5 * einsum('jk,lmcb,ci,ablm->ijka', kd[o, o], l2, t1,
                                     t2,
                                     optimize=['einsum_path', (1, 3), (1, 2),
                                               (0, 1)])

    #	 -0.5000 d(i,k)*l2(l,m,c,b)*t1(a,m)*t2(c,b,l,j)
    tpdm[o, o, o, v] += -0.5 * einsum('ik,lmcb,am,cblj->ijka', kd[o, o], l2, t1,
                                      t2,
                                      optimize=['einsum_path', (1, 3), (1, 2),
                                                (0, 1)])

    #	 -0.5000 d(i,k)*l2(l,m,c,b)*t1(c,j)*t2(a,b,l,m)
    tpdm[o, o, o, v] += -0.5 * einsum('ik,lmcb,cj,ablm->ijka', kd[o, o], l2, t1,
                                      t2,
                                      optimize=['einsum_path', (1, 3), (1, 2),
                                                (0, 1)])

    #	 -0.5000 P(i,j)l2(l,k,c,b)*t1(a,j)*t2(c,b,l,i)
    contracted_intermediate = -0.5 * einsum('lkcb,aj,cbli->ijka', l2, t1, t2,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    tpdm[o, o, o, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijka->jika', contracted_intermediate)

    #	 -0.5000 l2(l,k,c,b)*t1(a,l)*t2(c,b,i,j)
    tpdm[o, o, o, v] += -0.5 * einsum('lkcb,al,cbij->ijka', l2, t1, t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(i,j)l2(l,k,c,b)*t1(c,j)*t2(a,b,l,i)
    contracted_intermediate = 1.0 * einsum('lkcb,cj,abli->ijka', l2, t1, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    tpdm[o, o, o, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijka->jika', contracted_intermediate)

    #	  1.0000 l2(l,k,c,b)*t1(a,l)*t1(c,j)*t1(b,i)
    tpdm[o, o, o, v] += 1.0 * einsum('lkcb,al,cj,bi->ijka', l2, t1, t1, t1,
                                     optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #    D2(i,j,a,l):

    #	  1.0000 d(j,l)*t1(a,i)
    tpdm[o, o, v, o] += 1.0 * einsum('jl,ai->ijal', kd[o, o], t1)

    #	 -1.0000 d(i,l)*t1(a,j)
    tpdm[o, o, v, o] += -1.0 * einsum('il,aj->ijal', kd[o, o], t1)

    #	 -1.0000 d(j,l)*l1(k,b)*t2(a,b,k,i)
    tpdm[o, o, v, o] += -1.0 * einsum('jl,kb,abki->ijal', kd[o, o], l1, t2,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 d(i,l)*l1(k,b)*t2(a,b,k,j)
    tpdm[o, o, v, o] += 1.0 * einsum('il,kb,abkj->ijal', kd[o, o], l1, t2,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 l1(l,b)*t2(a,b,i,j)
    tpdm[o, o, v, o] += -1.0 * einsum('lb,abij->ijal', l1, t2)

    #	 -1.0000 d(j,l)*l1(k,b)*t1(a,k)*t1(b,i)
    tpdm[o, o, v, o] += -1.0 * einsum('jl,kb,ak,bi->ijal', kd[o, o], l1, t1, t1,
                                      optimize=['einsum_path', (1, 2), (1, 2),
                                                (0, 1)])

    #	  1.0000 d(i,l)*l1(k,b)*t1(a,k)*t1(b,j)
    tpdm[o, o, v, o] += 1.0 * einsum('il,kb,ak,bj->ijal', kd[o, o], l1, t1, t1,
                                     optimize=['einsum_path', (1, 2), (1, 2),
                                               (0, 1)])

    #	  1.0000 P(i,j)l1(l,b)*t1(a,j)*t1(b,i)
    contracted_intermediate = 1.0 * einsum('lb,aj,bi->ijal', l1, t1, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    tpdm[o, o, v, o] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijal->jial', contracted_intermediate)

    #	 -0.5000 d(j,l)*l2(k,m,c,b)*t1(a,m)*t2(c,b,k,i)
    tpdm[o, o, v, o] += -0.5 * einsum('jl,kmcb,am,cbki->ijal', kd[o, o], l2, t1,
                                      t2,
                                      optimize=['einsum_path', (1, 3), (1, 2),
                                                (0, 1)])

    #	 -0.5000 d(j,l)*l2(k,m,c,b)*t1(c,i)*t2(a,b,k,m)
    tpdm[o, o, v, o] += -0.5 * einsum('jl,kmcb,ci,abkm->ijal', kd[o, o], l2, t1,
                                      t2,
                                      optimize=['einsum_path', (1, 3), (1, 2),
                                                (0, 1)])

    #	  0.5000 d(i,l)*l2(k,m,c,b)*t1(a,m)*t2(c,b,k,j)
    tpdm[o, o, v, o] += 0.5 * einsum('il,kmcb,am,cbkj->ijal', kd[o, o], l2, t1,
                                     t2,
                                     optimize=['einsum_path', (1, 3), (1, 2),
                                               (0, 1)])

    #	  0.5000 d(i,l)*l2(k,m,c,b)*t1(c,j)*t2(a,b,k,m)
    tpdm[o, o, v, o] += 0.5 * einsum('il,kmcb,cj,abkm->ijal', kd[o, o], l2, t1,
                                     t2,
                                     optimize=['einsum_path', (1, 3), (1, 2),
                                               (0, 1)])

    #	  0.5000 P(i,j)l2(k,l,c,b)*t1(a,j)*t2(c,b,k,i)
    contracted_intermediate = 0.5 * einsum('klcb,aj,cbki->ijal', l2, t1, t2,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    tpdm[o, o, v, o] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijal->jial', contracted_intermediate)

    #	  0.5000 l2(k,l,c,b)*t1(a,k)*t2(c,b,i,j)
    tpdm[o, o, v, o] += 0.5 * einsum('klcb,ak,cbij->ijal', l2, t1, t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 P(i,j)l2(k,l,c,b)*t1(c,j)*t2(a,b,k,i)
    contracted_intermediate = -1.0 * einsum('klcb,cj,abki->ijal', l2, t1, t2,
                                            optimize=['einsum_path', (0, 1),
                                                      (0, 1)])
    tpdm[o, o, v, o] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijal->jial', contracted_intermediate)

    #	 -1.0000 l2(k,l,c,b)*t1(a,k)*t1(c,j)*t1(b,i)
    tpdm[o, o, v, o] += -1.0 * einsum('klcb,ak,cj,bi->ijal', l2, t1, t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 2),
                                                (0, 1)])

    #    D2(i,a,k,l):

    #	 -1.0000 d(i,l)*l1(k,a)
    tpdm[o, v, o, o] += -1.0 * einsum('il,ka->iakl', kd[o, o], l1)

    #	  1.0000 d(i,k)*l1(l,a)
    tpdm[o, v, o, o] += 1.0 * einsum('ik,la->iakl', kd[o, o], l1)

    #	  1.0000 l2(k,l,a,b)*t1(b,i)
    tpdm[o, v, o, o] += 1.0 * einsum('klab,bi->iakl', l2, t1)

    #    D2(a,j,k,l):

    #	  1.0000 d(j,l)*l1(k,a)
    tpdm[v, o, o, o] += 1.0 * einsum('jl,ka->ajkl', kd[o, o], l1)

    #	 -1.0000 d(j,k)*l1(l,a)
    tpdm[v, o, o, o] += -1.0 * einsum('jk,la->ajkl', kd[o, o], l1)

    #	 -1.0000 l2(k,l,a,b)*t1(b,j)
    tpdm[v, o, o, o] += -1.0 * einsum('klab,bj->ajkl', l2, t1)

    #    D2(a,b,c,d):

    #	  0.5000 l2(i,j,a,b)*t2(c,d,i,j)
    tpdm[v, v, v, v] += 0.5 * einsum('ijab,cdij->abcd', l2, t2)

    #	 -1.0000 l2(i,j,a,b)*t1(c,j)*t1(d,i)
    tpdm[v, v, v, v] += -1.0 * einsum('ijab,cj,di->abcd', l2, t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #    D2(a,b,c,i):

    #	  1.0000 l2(j,i,a,b)*t1(c,j)
    tpdm[v, v, v, o] += 1.0 * einsum('jiab,cj->abci', l2, t1)

    #    D2(a,b,i,d):

    #	 -1.0000 l2(j,i,a,b)*t1(d,j)
    tpdm[v, v, o, v] += -1.0 * einsum('jiab,dj->abid', l2, t1)

    #    D2(i,b,c,d):

    #	 -1.0000 l1(j,b)*t2(c,d,j,i)
    tpdm[o, v, v, v] += -1.0 * einsum('jb,cdji->ibcd', l1, t2)

    #	  1.0000 P(c,d)l1(j,b)*t1(c,i)*t1(d,j)
    contracted_intermediate = 1.0 * einsum('jb,ci,dj->ibcd', l1, t1, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    tpdm[o, v, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ibcd->ibdc', contracted_intermediate)

    #	  0.5000 P(c,d)l2(j,k,b,a)*t1(c,i)*t2(d,a,j,k)
    contracted_intermediate = 0.5 * einsum('jkba,ci,dajk->ibcd', l2, t1, t2,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    tpdm[o, v, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ibcd->ibdc', contracted_intermediate)

    #	 -1.0000 P(c,d)l2(j,k,b,a)*t1(c,k)*t2(d,a,j,i)
    contracted_intermediate = -1.0 * einsum('jkba,ck,daji->ibcd', l2, t1, t2,
                                            optimize=['einsum_path', (0, 1),
                                                      (0, 1)])
    tpdm[o, v, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ibcd->ibdc', contracted_intermediate)

    #	  0.5000 l2(j,k,b,a)*t1(a,i)*t2(c,d,j,k)
    tpdm[o, v, v, v] += 0.5 * einsum('jkba,ai,cdjk->ibcd', l2, t1, t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 l2(j,k,b,a)*t1(c,k)*t1(d,j)*t1(a,i)
    tpdm[o, v, v, v] += -1.0 * einsum('jkba,ck,dj,ai->ibcd', l2, t1, t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 2),
                                                (0, 1)])

    #    D2(a,i,c,d):

    #	  1.0000 l1(j,a)*t2(c,d,j,i)
    tpdm[v, o, v, v] += 1.0 * einsum('ja,cdji->aicd', l1, t2)

    #	 -1.0000 P(c,d)l1(j,a)*t1(c,i)*t1(d,j)
    contracted_intermediate = -1.0 * einsum('ja,ci,dj->aicd', l1, t1, t1,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    tpdm[v, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'aicd->aidc', contracted_intermediate)

    #	 -0.5000 P(c,d)l2(j,k,a,b)*t1(c,i)*t2(d,b,j,k)
    contracted_intermediate = -0.5 * einsum('jkab,ci,dbjk->aicd', l2, t1, t2,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    tpdm[v, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'aicd->aidc', contracted_intermediate)

    #	  1.0000 P(c,d)l2(j,k,a,b)*t1(c,k)*t2(d,b,j,i)
    contracted_intermediate = 1.0 * einsum('jkab,ck,dbji->aicd', l2, t1, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    tpdm[v, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'aicd->aidc', contracted_intermediate)

    #	 -0.5000 l2(j,k,a,b)*t1(b,i)*t2(c,d,j,k)
    tpdm[v, o, v, v] += -0.5 * einsum('jkab,bi,cdjk->aicd', l2, t1, t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 l2(j,k,a,b)*t1(c,k)*t1(d,j)*t1(b,i)
    tpdm[v, o, v, v] += 1.0 * einsum('jkab,ck,dj,bi->aicd', l2, t1, t1, t1,
                                     optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #    D2(i,j,a,b):

    #	  1.0000 t2(a,b,i,j)
    tpdm[o, o, v, v] += 1.0 * einsum('abij->ijab', t2)

    #	 -1.0000 P(i,j)t1(a,j)*t1(b,i)
    contracted_intermediate = -1.0 * einsum('aj,bi->ijab', t1, t1)
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)l1(k,c)*t1(a,j)*t2(b,c,k,i)
    contracted_intermediate = 1.0 * einsum('kc,aj,bcki->ijab', l1, t1, t2,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate) + -1.00000 * einsum('ijab->ijba',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'ijab->jiba', contracted_intermediate)

    #	  1.0000 P(a,b)l1(k,c)*t1(a,k)*t2(b,c,i,j)
    contracted_intermediate = 1.0 * einsum('kc,ak,bcij->ijab', l1, t1, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->ijba', contracted_intermediate)

    #	  1.0000 P(i,j)l1(k,c)*t1(c,j)*t2(a,b,k,i)
    contracted_intermediate = 1.0 * einsum('kc,cj,abki->ijab', l1, t1, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate)

    #	 -0.5000 P(i,j)l2(k,l,d,c)*t2(a,b,l,j)*t2(d,c,k,i)
    contracted_intermediate = -0.5 * einsum('kldc,ablj,dcki->ijab', l2, t2, t2,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate)

    #	  0.2500 l2(k,l,d,c)*t2(a,b,k,l)*t2(d,c,i,j)
    tpdm[o, o, v, v] += 0.25 * einsum('kldc,abkl,dcij->ijab', l2, t2, t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 l2(k,l,d,c)*t2(a,d,i,j)*t2(b,c,k,l)
    tpdm[o, o, v, v] += -0.5 * einsum('kldc,adij,bckl->ijab', l2, t2, t2,
                                      optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 P(i,j)l2(k,l,d,c)*t2(a,d,l,j)*t2(b,c,k,i)
    contracted_intermediate = 1.0 * einsum('kldc,adlj,bcki->ijab', l2, t2, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate)

    #	 -0.5000 l2(k,l,d,c)*t2(a,d,k,l)*t2(b,c,i,j)
    tpdm[o, o, v, v] += -0.5 * einsum('kldc,adkl,bcij->ijab', l2, t2, t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)l1(k,c)*t1(a,j)*t1(b,k)*t1(c,i)
    contracted_intermediate = 1.0 * einsum('kc,aj,bk,ci->ijab', l1, t1, t1, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (1, 2), (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate) + -1.00000 * einsum('ijab->ijba',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'ijab->jiba', contracted_intermediate)

    #	  0.5000 P(i,j)*P(a,b)l2(k,l,d,c)*t1(a,j)*t1(b,l)*t2(d,c,k,i)
    contracted_intermediate = 0.5 * einsum('kldc,aj,bl,dcki->ijab', l2, t1, t1,
                                           t2, optimize=['einsum_path', (0, 3),
                                                         (1, 2), (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate) + -1.00000 * einsum('ijab->ijba',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'ijab->jiba', contracted_intermediate)

    #	  0.5000 P(i,j)*P(a,b)l2(k,l,d,c)*t1(a,j)*t1(d,i)*t2(b,c,k,l)
    contracted_intermediate = 0.5 * einsum('kldc,aj,di,bckl->ijab', l2, t1, t1,
                                           t2, optimize=['einsum_path', (0, 3),
                                                         (1, 2), (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate) + -1.00000 * einsum('ijab->ijba',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'ijab->jiba', contracted_intermediate)

    #	 -0.5000 l2(k,l,d,c)*t1(a,l)*t1(b,k)*t2(d,c,i,j)
    tpdm[o, o, v, v] += -0.5 * einsum('kldc,al,bk,dcij->ijab', l2, t1, t1, t2,
                                      optimize=['einsum_path', (0, 1), (0, 2),
                                                (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)l2(k,l,d,c)*t1(a,l)*t1(d,j)*t2(b,c,k,i)
    contracted_intermediate = 1.0 * einsum('kldc,al,dj,bcki->ijab', l2, t1, t1,
                                           t2, optimize=['einsum_path', (0, 1),
                                                         (0, 2), (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate) + -1.00000 * einsum('ijab->ijba',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'ijab->jiba', contracted_intermediate)

    #	 -0.5000 l2(k,l,d,c)*t1(d,j)*t1(c,i)*t2(a,b,k,l)
    tpdm[o, o, v, v] += -0.5 * einsum('kldc,dj,ci,abkl->ijab', l2, t1, t1, t2,
                                      optimize=['einsum_path', (0, 1), (0, 2),
                                                (0, 1)])

    #	  1.0000 l2(k,l,d,c)*t1(a,l)*t1(b,k)*t1(d,j)*t1(c,i)
    tpdm[o, o, v, v] += 1.0 * einsum('kldc,al,bk,dj,ci->ijab', l2, t1, t1, t1,
                                     t1,
                                     optimize=['einsum_path', (0, 1), (0, 3),
                                               (0, 2), (0, 1)])

    #    D2(a,b,i,j):

    #	  1.0000 l2(i,j,a,b)
    tpdm[v, v, o, o] += 1.0 * einsum('ijab->abij', l2)

    #    D2(i,a,j,b):

    #	  1.0000 d(i,j)*l1(k,a)*t1(b,k)
    tpdm[o, v, o, v] += 1.0 * einsum('ij,ka,bk->iajb', kd[o, o], l1, t1,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 l1(j,a)*t1(b,i)
    tpdm[o, v, o, v] += -1.0 * einsum('ja,bi->iajb', l1, t1)

    #	  0.5000 d(i,j)*l2(k,l,a,c)*t2(b,c,k,l)
    tpdm[o, v, o, v] += 0.5 * einsum('ij,klac,bckl->iajb', kd[o, o], l2, t2,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 l2(k,j,a,c)*t2(b,c,k,i)
    tpdm[o, v, o, v] += -1.0 * einsum('kjac,bcki->iajb', l2, t2)

    #	 -1.0000 l2(k,j,a,c)*t1(b,k)*t1(c,i)
    tpdm[o, v, o, v] += -1.0 * einsum('kjac,bk,ci->iajb', l2, t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #    D2(a,i,j,b):

    #	 -1.0000 d(i,j)*l1(k,a)*t1(b,k)
    tpdm[v, o, o, v] += -1.0 * einsum('ij,ka,bk->aijb', kd[o, o], l1, t1,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 l1(j,a)*t1(b,i)
    tpdm[v, o, o, v] += 1.0 * einsum('ja,bi->aijb', l1, t1)

    #	 -0.5000 d(i,j)*l2(k,l,a,c)*t2(b,c,k,l)
    tpdm[v, o, o, v] += -0.5 * einsum('ij,klac,bckl->aijb', kd[o, o], l2, t2,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 l2(k,j,a,c)*t2(b,c,k,i)
    tpdm[v, o, o, v] += 1.0 * einsum('kjac,bcki->aijb', l2, t2)

    #	  1.0000 l2(k,j,a,c)*t1(b,k)*t1(c,i)
    tpdm[v, o, o, v] += 1.0 * einsum('kjac,bk,ci->aijb', l2, t1, t1,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #    D2(i,a,b,j):

    #	 -1.0000 d(i,j)*l1(k,a)*t1(b,k)
    tpdm[o, v, v, o] += -1.0 * einsum('ij,ka,bk->iabj', kd[o, o], l1, t1,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 l1(j,a)*t1(b,i)
    tpdm[o, v, v, o] += 1.0 * einsum('ja,bi->iabj', l1, t1)

    #	 -0.5000 d(i,j)*l2(k,l,a,c)*t2(b,c,k,l)
    tpdm[o, v, v, o] += -0.5 * einsum('ij,klac,bckl->iabj', kd[o, o], l2, t2,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 l2(k,j,a,c)*t2(b,c,k,i)
    tpdm[o, v, v, o] += 1.0 * einsum('kjac,bcki->iabj', l2, t2)

    #	  1.0000 l2(k,j,a,c)*t1(b,k)*t1(c,i)
    tpdm[o, v, v, o] += 1.0 * einsum('kjac,bk,ci->iabj', l2, t1, t1,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #    D2(a,i,b,j):

    #	  1.0000 d(i,j)*l1(k,a)*t1(b,k)
    tpdm[v, o, v, o] += 1.0 * einsum('ij,ka,bk->aibj', kd[o, o], l1, t1,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 l1(j,a)*t1(b,i)
    tpdm[v, o, v, o] += -1.0 * einsum('ja,bi->aibj', l1, t1)

    #	  0.5000 d(i,j)*l2(k,l,a,c)*t2(b,c,k,l)
    tpdm[v, o, v, o] += 0.5 * einsum('ij,klac,bckl->aibj', kd[o, o], l2, t2,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 l2(k,j,a,c)*t2(b,c,k,i)
    tpdm[v, o, v, o] += -1.0 * einsum('kjac,bcki->aibj', l2, t2)

    #	 -1.0000 l2(k,j,a,c)*t1(b,k)*t1(c,i)
    tpdm[v, o, v, o] += -1.0 * einsum('kjac,bk,ci->aibj', l2, t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    return tpdm




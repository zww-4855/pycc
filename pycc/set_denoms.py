import numpy as np

def D1denomFast(epsaa,epsbb,occ_aa,occ_bb,virt_aa,virt_bb,n):
    D1_aa = 1.0/ (-epsaa[virt_aa,n] + epsaa[n,occ_aa])
    D1_bb = 1.0/ (-epsbb[virt_bb,n] + epsbb[n,occ_bb])
    
    D1_aa=D1_aa.transpose(1,0)
    D1_bb=D1_bb.transpose(1,0)
    return D1_aa, D1_bb

def D2denomFast(epsaa,epsbb,occ_aa,occ_bb,virt_aa,virt_bb,n):
    D2_bb = 1.0 / (
        -epsbb[virt_bb, n, n, n]
        - epsbb[n, virt_bb, n, n]
        + epsbb[n, n, occ_bb, n]
        + epsbb[n, n, n, occ_bb]
    )
    D2_ab = 1.0 / (
        -epsaa[virt_aa, n, n, n]
        - epsbb[n, virt_bb, n, n]
        + epsaa[n, n, occ_aa, n]
        + epsbb[n, n, n, occ_bb]
    )

    D2_aa = 1.0 / (
        -epsaa[virt_aa, n, n, n]
        - epsaa[n, virt_aa, n, n]
        + epsaa[n, n, occ_aa, n]
        + epsaa[n, n, n, occ_aa]
    )


    D2_aa = D2_aa.transpose(2,3,0,1)
    D2_bb = D2_bb.transpose(2,3,0,1)
    D2_ab = D2_ab.transpose(2,3,0,1)

    return D2_aa, D2_bb, D2_ab

def D3denomFast(epsaa,epsbb,occ_aa,occ_bb,virt_aa,virt_bb,n):
    D3_aaa = 1.0 / (
        -epsaa[virt_aa, n, n, n, n, n]
        - epsaa[n, virt_aa, n, n, n, n]
        - epsaa[n,      n, virt_aa,n, n, n]
        + epsaa[n, n, n ,occ_aa,n, n]
        + epsaa[n, n, n, n, occ_aa, n]
        + epsaa[n, n, n, n, n, occ_aa]
    )

    D3_bbb = 1.0 / (
        -epsbb[virt_bb, n, n, n, n, n]
        - epsbb[n, virt_bb, n, n, n, n]
        - epsbb[n,      n, virt_bb,n, n, n]
        + epsbb[n, n, n ,occ_bb,n, n]
        + epsbb[n, n, n, n, occ_bb, n]
        + epsbb[n, n, n, n, n, occ_bb]
    )

    D3_aab = 1.0 / (
        -epsaa[virt_aa, n, n, n, n, n]
        - epsaa[n, virt_aa, n, n, n, n]
        - epsbb[n,      n, virt_bb,n, n, n]
        + epsaa[n, n, n ,occ_aa,n, n]
        + epsaa[n, n, n, n, occ_aa, n]
        + epsbb[n, n, n, n, n, occ_bb]
    )

    D3_abb = 1.0 / (
        -epsaa[virt_aa, n, n, n, n, n]
        - epsbb[n, virt_bb, n, n, n, n]
        - epsbb[n,      n, virt_bb,n, n, n]
        + epsaa[n, n, n ,occ_aa,n, n]
        + epsbb[n, n, n, n, occ_bb, n]
        + epsbb[n, n, n, n, n, occ_bb]
    )

    D3_aaa = D3_aaa.transpose(3,4,5,0,1,2)
    D3_bbb = D3_bbb.transpose(3,4,5,0,1,2)
    D3_aab = D3_aab.transpose(3,4,5,0,1,2)
    D3_abb = D3_abb.transpose(3,4,5,0,1,2)
    return D3_aaa, D3_bbb, D3_aab, D3_abb


def D1denomSlow(epsaa,occ_aa,virt_aa,n):
    D1=1.0/(-epsaa[virt_aa,n]+epsaa[n,occ_aa])
    D1=D1.transpose(1,0)
    return D1

def D2denomSlow(epsaa,occ_aa,virt_aa,n):
    D2=1.0 / (
        -epsaa[virt_aa, n, n, n]
        - epsaa[n, virt_aa, n, n]
        + epsaa[n, n, occ_aa, n]
        + epsaa[n, n, n, occ_aa]
    )
    D2=D2.transpose(2,3,0,1)
    return D2

import numpy as np


def D3denomSlow(epsaa, occ_aa, virt_aa, n, omega=0.0, level_shift=0.0):
    """
    Build the inverse triples denominator

        D3^{-1}_{ijkabc}
        =
        1 / (eps_i + eps_j + eps_k - eps_a - eps_b - eps_c + omega)

    where omega is the EOM-UCCSD excitation energy.

    Parameters
    ----------
    epsaa : ndarray
        Broadcastable orbital-energy tensor.

    occ_aa : ndarray or list
        Occupied spin-orbital indices.

    virt_aa : ndarray or list
        Virtual spin-orbital indices.

    n : int or slice
        Broadcast helper index used in your current epsaa construction.

    omega : float, optional
        EOM-UCCSD excitation energy used to offset the triples denominator.
        Default is 0.0, which recovers the original denominator.

    level_shift : float, optional
        Optional additional denominator shift for numerical stability.

    Returns
    -------
    D3 : ndarray
        Inverse triples denominator with shape ordered as (i, j, k, a, b, c).
    """

    omega = float(omega)
    level_shift = float(level_shift)

    denom = (
        - epsaa[virt_aa, n,       n,       n,      n,      n]
        - epsaa[n,       virt_aa, n,       n,      n,      n]
        - epsaa[n,       n,       virt_aa, n,      n,      n]
        + epsaa[n,       n,       n,       occ_aa, n,      n]
        + epsaa[n,       n,       n,       n,      occ_aa, n]
        + epsaa[n,       n,       n,       n,      n,      occ_aa]
    )

    # EOM-style shift:
    # denominator = eps_i + eps_j + eps_k - eps_a - eps_b - eps_c + omega
    denom = denom + omega + level_shift

    D3 = 1.0 / denom

    # Original ordering transformation
    D3 = D3.transpose(3, 4, 5, 0, 1, 2)

    print("shapes:", np.shape(D3), occ_aa, virt_aa, n, np.shape(epsaa))

    return D3


def D4denomSlow(epsaa,occ_aa,virt_aa,n):
    D4=1.0/(
            -epsaa[virt_aa, n, n, n, n, n, n, n]
           -epsaa[n,      virt_aa, n, n, n, n, n, n]
           -epsaa[n, n,           virt_aa, n, n, n, n, n]
           -epsaa[n, n, n,                virt_aa, n, n, n, n]
           +epsaa[n, n, n, n, occ_aa, n, n, n]
           +epsaa[n, n, n, n, n,       occ_aa, n, n]
           +epsaa[n, n, n, n, n, n,            occ_aa, n]
           +epsaa[n, n, n, n, n, n, n,                 occ_aa] )
    D4=D4.transpose(4,5,6,7,0,1,2,3)
    return D4

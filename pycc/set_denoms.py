
def D1denomFast(epsaa,epsbb,occ_aa,occ_bb,virt_aa,virt_bb,n):
    D1_aa = 1.0/ (-epsaa[virt_aa,n] + epsaa[n,occ_aa])
    D1_bb = 1.0/ (-epsbb[virt_bb,n] + epsaa[n,occ_bb])
    
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

def D3denomSlow(epsaa,occ_aa,virt_aa,n):
    D3 = 1.0/(
            -epsaa[virt_aa,n,n,n,n,n]
            -epsaa[n,virt_aa,n,n,n,n]
            -epsaa[n,n,virt_aa,n,n,n]
            +epsaa[n,n,n,occ_aa,n,n]
            +epsaa[n,n,n,n,occ_aa,n]
            +epsaa[n,n,n,n,n,occ_aa] )
    D3=D3.transpose(3,4,5,0,1,2)
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

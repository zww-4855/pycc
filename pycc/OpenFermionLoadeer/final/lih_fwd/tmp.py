n_spin_orbitals = 12
nocc=2
alpha_case = list(range(0,n_spin_orbitals,2))
beta_case = list(range(1,n_spin_orbitals,2))

def add_pure_spinT2_amps(occ_list,virt_list):
    for II in occ_list:
        occ_idx = occ_list.index(II)+1
        for JJ in occ_list[occ_idx:]:
            for AA in virt_list:
                virt_idx = virt_list.index(AA)+1
                for BB in virt_list[virt_idx:]:
                    print("a^b^ji:",AA,BB,II,JJ)


add_pure_spinT2_amps(alpha_case[:nocc],alpha_case[nocc:])
nvirt=4
for I in range(nocc):
    i=2*I
    for J in range(nocc):
        j=2*J+1
        for A in range(nvirt):
            a=2*nocc+2*A
            for B in range(nvirt):
                b=2*nocc+2*B+1
                spin_bool = (i % 2) == (j % 2) == (a % 2) == (b % 2)
                if spin_bool:
                    print("a^b^ji:",a,b,j,i)


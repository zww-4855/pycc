from itertools import product, combinations

def generate_excitation_indices(n_qubits, n_electrons):
    occ = list(range(n_electrons))
    virt = list(range(n_electrons, n_qubits))

#    singles = list(product(occ, virt))        # (i, a)

    same_spin_doubles = [(a, b, i,j)
                         for (i, j) in combinations(occ, 2)
                         for (a, b) in combinations(virt, 2)
                         if (i % 2) == (j % 2) == (a % 2) == (b % 2)]

    alpha = [d for d in same_spin_doubles if d[0] % 2 == 0]  # i is alpha
    beta  = [d for d in same_spin_doubles if d[0] % 2 == 1]  # i is beta
    
    # interleave them
    doubles_alt = [x for pair in zip(alpha, beta) for x in pair]
    
    print("doubles_alt:",doubles_alt)

    return doubles_alt

generate_excitation_indices(12,4)

def print_mixed_spin(n_qubits,n_electrons):
    nocc = int(n_electrons/2)
    nvirt = int(int(n_qubits/2) - nocc)

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
                    print("a^b^ji:",a,b,j,i)

print_mixed_spin(12,4)

def print_singles(n_qubits, n_electrons):
    occ = list(range(n_electrons))
    virt = list(range(n_electrons, n_qubits))

    alpha=[]
    beta=[]
    total=[]
    for i in occ:
        for a in virt:
            if (a % 2) != (i % 2):
                # spin mismatch => would flip spin, skip
                continue
            print("a^ i",a,i)
            total.append((a,i))

    alpha = [d for d in total if d[0] % 2 == 0]  # i is alpha
    beta = [d for d in total if d[0] % 2 == 1]
    # interleave them
    singles_alt = [x for pair in zip(alpha, beta) for x in pair]
    print("singles alt:",singles_alt)


print_singles(12,4)

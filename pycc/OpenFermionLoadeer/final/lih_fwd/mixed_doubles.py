from itertools import combinations

# occ = occupied spin-orbital indices
# virt = virtual spin-orbital indices
occ = list(range(4))
virt = list(range(4, 12))
mixed_ab_doubles = [
    (i, j, a, b)
    for (i, j) in combinations(occ, 2)
    for (a, b) in combinations(virt, 2)
    if (i % 2 != j % 2)   # ensure one alpha, one beta occupied
    and (a % 2 != b % 2)  # ensure one alpha, one beta virtual
]
print("mixed_ab_doubles:",mixed_ab_doubles)
alpha_first = [d for d in mixed_ab_doubles if d[0] % 2 == 0]  # i=α, j=β
beta_first  = [d for d in mixed_ab_doubles if d[0] % 2 == 1]  # i=β, j=α

# Interleave
mixed_alt = [x for pair in zip(alpha_first, beta_first) for x in pair]
#print("mixed_alt:",mixed_alt)
# Append leftovers
if len(alpha_first) > len(beta_first):
    mixed_alt.extend(alpha_first[len(beta_first):])
else:
    mixed_alt.extend(beta_first[len(alpha_first):])



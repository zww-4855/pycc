Available methods based on CC and MBPT
======================================

The following is a list of currently supported, spin-orbital based traditional and unitary CC (UCC) schemes:

Traditional:
* CCD
* CCSD
* CCSDT
* CCSDTq, where 'q' refers to the factorized, 5th and 6th order quadruples energy corrections (Qf)

Unitary:
* UCC3, aka LCCD
* UCCD4, Fourth-order UCC functional comprised of only T2 operators
* UCCSD4, Fourth-order UCC functional comprised of both T1 and T2 operators
* UCCD5, Fifth-order UCC functional comprised of only T2 operators
* UCCSD5, Fifth-orer UCC functional comprised of both T1 and T2 operators

To use any of these methods, simply provide the string - as written - to be the value of the "slowSOcalc" key in the cc_info dictionary. As an example to run a CCSDTq calculation, which performs traditional CCSDT then performs a post hoc energy correction - based on the factorization theorem - to account for missing quadruple excitations:


>>> atomString=f'H 0. 0. 0.0; F 0.0 0.0 1.128'
>>> basis='dz'
>>> pyscf_mol = pyscf.M(
>>>    atom=atomString,
>>>    verbose=5,
>>>    symmetry =True,
>>>    basis=basis)
>>> pyscf_mf = pyscf_mol.RHF(pyscf_mol)
>>> pyscf_mf.run()
>>> cc_info = {"slowSOcalc":"CCSDTq"}
>>> pycc_obj = pycc.DriveCC(pyscf_mf,pyscf_mol,pycc_info)
>>> pycc_obj.kernel(cc_info)

Customizing the CC calculation
==============================
Other information can be added to the cc_info dictionary for use by pycc. Using an UCCSD5 calculation as a reference point,

>>> cc_info={"slowSOcalc":"UCCSD5","stopping_eps":10**-8,"diis_size":5,"diis_start_cycle":2,"dropcore":1,"dump_tamps":False,"max_iter":100}

We see that the above keywords assert that:
* The lowest-energy molecular orbital electrons are requested to be dropped from the correlation calculation
* The stopping criteria for the CC iterations is an energy difference < 10**-8
* There is a maximum of 5 DIIS vectors spanning the extrapolation subspace, which begin accumulating after the 2nd CC iteration
* We have requested that the T2 amplitudes *not* be dumped to a pickle file upon convergence
* There are a maximum of 100 CC iterations allowed for this calculation.  


If these keys are not set, appropriate defaults are used instead. 


Interfaces with other software
==============================



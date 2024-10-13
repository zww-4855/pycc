Getting Started
===============


pycc functionality depends upon a prior PySCF self-consistent field (SCF) calculation. Unless specifically mentioned otherwise, pycc can be used in conjunction with RHF or UHF orbitals. After installation of PySCF, a mean-field object can be initialized using the following generic script:


>>> import pyscf
>>> atomString = 'H 0 0 0; F 0 0 0.917'
>>> pyscf_mol = pyscf.M(
>>>     atom=atomString,
>>>     verbose=5,
>>>     basis='cc-pvdz')
>>> pyscf_mf = pyscf_mol.RHF()
>>> pyscf_mf.conv_tol_grad=1E-10
>>> pyscf_mf.run()

and run until convergence of the SCF procedure. Once this is successfully dont, the pertient details regarding the SCF - like orbitals, Hartree-Fock energy, etc - are required by pycc. Consequently, intantiating an object of the DriveCC() class relies, in tern, on the pyscf_mf and pyscf_mol objects, as well as a dictionary, 'cc_info' containing pertinent details regarding customization of the CC calculation. 


>>> import pycc
>>> cc_info = {"slowSOcalc":"UCCD5","stopping_eps":10**-8,"diis_size":5,"diis_start_cycle":2, dropcore":1}
>>> pycc_obj=pycc.DriveCC(pyscf_mf,pyscf_mol,cc_info)

More details will be added that fully define the possible paramters defining 'cc_info'. 

To iterate the CC equations to convergence, 

>>> pyscf_obj.kernel(cc_info)

Upon convergence of the residual equations, pertinent information regarding the calculation can be manipulated using the DriveCC object 'pyscf_obj'. For example, the converged set of T2 amplitudes can be extracted like

>>> pyscf_obj.tamps["t2aa"]



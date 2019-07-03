import scipy
import openfermion
import openfermionpsi4
import os
import numpy as np
import copy
import random 
import sys
import copy,itertools
import time 

import operator_pools
import vqe_methods
from tVQE import *

from openfermion import *

from openfermionprojectq import *
from projectq.ops import X, All, Measure
from projectq.backends import CommandPrinter, CircuitDrawer


def adapt_vqe(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 1,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-3,
        theta_thresh    = 1e-7,
        adapt_maxiter   = 200,
        pool            = operator_pools.singlet_GSD(),
        reference       = 'rhf',
        brueckner       = 0,
        ref_state       = None,
        spin_adapt      = True,
        fci_nat_orb     = 0,
        cisd_nat_orb    = 0,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{
    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity, reference = reference)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=0, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_bccd = brueckner, # Brueckner CCD
                run_fci=1, 
                fci_no = fci_nat_orb,
                cisd_no = cisd_nat_orb,
                delete_input=1)
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    #print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    if brueckner == 1:
        print(' BCCD energy     %20.16f au' %(molecule.bccd_energy))
    if reference == 'rhf':
        print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    # if we are going to transform to FCI NOs, it doesn't make sense to transform to CISD NOs
    if cisd_nat_orb == 1 and fci_nat_orb == 0:
        print(' Basis transformed to the CISD natural orbitals')
    if fci_nat_orb == 1:
        print(' Basis transformed to the FCI natural orbitals')

    #Build p-h reference and map it to JW transform
    if ref_state == None:
        ref_state = list(range(0,molecule.n_electrons))

    reference_ket = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(ref_state, molecule.n_qubits)).transpose()
    reference_bra = reference_ket.transpose().conj()

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)

    #Thetas
    parameters = []

    pool.generate_SparseMatrix()
    pool.gradient_print_thresh = theta_thresh
    
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    
    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []
    curr_state = 1.0*reference_ket

    print(" Now start to grow the ansatz")
    for n_iter in range(0,adapt_maxiter):
    
        print("\n\n\n")
        print(" --------------------------------------------------------------------------")
        print("                         ADAPT-VQE iteration: ", n_iter)                 
        print(" --------------------------------------------------------------------------")
        next_index = None
        next_deriv = 0
        curr_norm = 0
        
        print(" Check each new operator for coupling")
        next_term = []
        print(" Measure Operator Pool Gradients:")
        sig = hamiltonian.dot(curr_state)
        e_curr = curr_state.T.conj().dot(sig)[0,0]
        var = sig.T.conj().dot(sig)[0,0] - e_curr**2
        uncertainty = np.sqrt(var.real)
        assert(np.isclose(var.imag,0))
        print(" Variance:    %12.8f" %var.real)
        print(" Uncertainty: %12.8f" %uncertainty)
        for oi in range(pool.n_ops):
            
            gi = pool.compute_gradient_i(oi, curr_state, sig)

            curr_norm += gi*gi
            if abs(gi) > abs(next_deriv):
                next_deriv = gi
                next_index = oi

        curr_norm = np.sqrt(curr_norm)

        min_options = {'gtol': theta_thresh, 'disp':False}
     
        max_of_gi = next_deriv
        print(" Norm of <[H,A]> = %12.8f" %curr_norm)
        print(" Max  of <[H,A]> = %12.8f" %max_of_gi)

        converged = False
        if adapt_conver == "norm":
            if curr_norm < adapt_thresh:
                converged = True
        elif adapt_conver == "var":
            if abs(var) < adapt_thresh:
                #variance
                converged = True
        else:
            print(" FAIL: Convergence criterion not defined")
            exit()

        if converged:
            print(" Ansatz Growth Converged!")
            print(" Number of operators in ansatz: ", len(ansatz_ops))
            print(" *Finished: %20.12f" % trial_model.curr_energy)
            print(" -----------Final ansatz----------- ")
            print(" %4s %12s %18s" %("#","Coeff","Term"))
            for si in range(len(ansatz_ops)):
                opstring = pool.get_string_for_term(ansatz_ops[si])
                print(" %4i %12.8f %s" %(si, parameters[si], opstring) )
            build_circuit(molecule.n_electrons, molecule.n_qubits, ansatz_ops, parameters[:len(ansatz_ops)])
            break
        
        print(" Add operator %4i" %next_index)
        parameters.insert(0,0)
        ansatz_ops.insert(0,pool.fermi_ops[next_index])
        ansatz_mat.insert(0,pool.spmat_ops[next_index])
        
        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)
        

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                options = min_options, method = 'BFGS', callback=trial_model.callback)
    
        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" -----------New ansatz----------- ")
        print(" %4s %12s %18s" %("#","Coeff","Term"))
        for si in range(len(ansatz_ops)):
            opstring = pool.get_string_for_term(ansatz_ops[si])
            print(" %4i %12.8f %s" %(si, parameters[si], opstring) )

# }}}

def ucc(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 1,
        theta_thresh    = 1e-7,
        pool            = operator_pools.singlet_GSD(),
        spin_adapt      = True,
        reference       = 'rhf',
        ref_state       = None,
        k               = 1, # k in k-UpCCGSD
        random_params   = False,
        random_all      = False,
        random_t1       = False,
        random_t2       = False,
        t1_first        = True,
        ops_order       = None,
        hf_stability    = 'none',
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{

    start_time = time.time()
    outfilename = psi4_filename[5:]
    #sys.stdout = open(outfilename, 'w')

    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity, reference = reference, hf_stability = hf_stability)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=0, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    #print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #Build p-h reference and map it to JW transform
    if ref_state == None:
        ref_state = list(range(0,molecule.n_electrons))

    reference_ket = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(ref_state, molecule.n_qubits)).transpose()
    reference_bra = reference_ket.transpose().conj()

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)

    #Thetas
    if random_params:
        parameters = np.random.rand(k*pool.n_ops)
    else:
        parameters = [0 for i in range(k*pool.n_ops)] 

    print(" Number of parameters: %s" % len(parameters))
    print(" Trotter-Suzuki order: %s" % k)
    print(" Initial parameters: %s" % parameters)
    pool.generate_SparseMatrix()
    
    n_t1 = int(pool.n_orb*(pool.n_orb-1)/2)
    n_t2 = pool.n_ops - n_t1

    if ops_order == None:

        if random_all: # randomize all operators with no particular order
            print(" All operators in the pool in random order.")

            ops_order = random.sample(range(pool.n_ops), pool.n_ops)

        else: # randomize (or not) within excitation rank and order

            if random_t1:
                print(" T1 operators in the pool in random order.")
                t1_order = random.sample(range(n_t1), n_t1)
            else:
                print(" T1 operators in the pool in lexical order.")
                t1_order = [i for i in range(n_t1)]

            if random_t2:
                print(" T2 operators in the pool in random order.")
                t2_order = random.sample(range(n_t1, pool.n_ops), n_t2)
            else:
                print(" T2 operators in the pool in lexical order.")
                t2_order = [i for i in range(n_t1, pool.n_ops)]

            if t1_first:
                print(" T1 first, then T2.")
                ops_order = t1_order + t2_order
            else:
                print(" T2 first, then T1.")
                ops_order = t2_order + t1_order

    print(' Operator order: %s\n' % ops_order)
    for i in ops_order:
        print(' Operator #%s: %s\n' % (i, pool.fermi_ops[i]))
    #sys.stdout = open(filename, 'w')

    #mat = [pool.spmat_ops[i] for i in ops_order]
    #ops = [pool.fermi_ops[i] for i in ops_order]
    mat = [copy.copy(pool.spmat_ops[i]) for _ in range(k) for i in ops_order]
    ops = [copy.copy(pool.fermi_ops[i]) for _ in range(k) for i in ops_order]

    ucc = tUCCSD(hamiltonian, mat, reference_ket, parameters, 1)
    #ucc = UCC(hamiltonian, pool.spmat_ops, reference_ket, parameters, k)
    
    opt_result = scipy.optimize.minimize(ucc.energy, 
                parameters, options = {'gtol': 1e-6, 'disp':True}, 
                method = 'BFGS', callback=ucc.callback)
    print(" Time elapsed: %5.3f s" % (time.time() - start_time))
    print(" Finished: %20.12f" % ucc.curr_energy)
    parameters = opt_result['x']
    for p in parameters:
        print(p)
    print( 'Warning: circuit below is for the present simulation!\n\n')
    build_circuit(molecule.n_electrons, molecule.n_qubits, ops, parameters)

    #filename = '_'.join(map(str, ops_order)) + '.out'
    cmd_string = 'mv '+outfilename + ' ' + '_'.join(map(str, ops_order)) + '.out'
    #os.system(cmd_string)

# }}}

def test_random(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 1,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-3,
        theta_thresh    = 1e-7,
        adapt_maxiter   = 200,
        pool            = operator_pools.singlet_GSD(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random(),
        seed            = 1
        ):

    # {{{
    random.seed(seed)

    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()
    reference_bra = reference_ket.transpose().conj()

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)

    #Thetas
    parameters = []

    pool.generate_SparseMatrix()
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    
    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []
    curr_state = 1.0*reference_ket

    print(" Now start to grow the ansatz")
    for n_iter in range(0,adapt_maxiter):
    
        print("\n\n\n")
        print(" --------------------------------------------------------------------------")
        print("                         ADAPT-VQE iteration: ", n_iter)                 
        print(" --------------------------------------------------------------------------")
        next_index = None
        next_deriv = 0
        curr_norm = 0
        
        print(" Check each new operator for coupling")
        next_term = []
        print(" Measure commutators:")
        sig = hamiltonian.dot(curr_state)
        for op_trial in range(pool.n_ops):
            
            opA = pool.spmat_ops[op_trial]
            com = 2*(curr_state.transpose().conj().dot(opA.dot(sig))).real
            assert(com.shape == (1,1))
            com = com[0,0]
            assert(np.isclose(com.imag,0))
            com = com.real
            opstring = ""
            for t in pool.fermi_ops[op_trial].terms:
                opstring += str(t)
                break
       
            if abs(com) > adapt_thresh:
                print(" %4i %40s %12.8f" %(op_trial, opstring, com) )

            curr_norm += com*com
            if abs(com) > abs(next_deriv):
                next_deriv = com
                next_index = op_trial

      
        next_index = random.choice(list(range(pool.n_ops)))
        curr_norm = np.sqrt(curr_norm)

        min_options = {'gtol': theta_thresh, 'disp':False}
     
        max_of_com = next_deriv
        print(" Norm of <[A,H]> = %12.8f" %curr_norm)
        print(" Max  of <[A,H]> = %12.8f" %max_of_com)

        converged = False
        if adapt_conver == "norm":
            if curr_norm < adapt_thresh:
                converged = True
        else:
            print(" FAIL: Convergence criterion not defined")
            exit()

        if converged:
            print(" Ansatz Growth Converged!")
            print(" Number of operators in ansatz: ", len(ansatz_ops))
            print(" *Finished: %20.12f" % trial_model.curr_energy)
            print(" -----------Final ansatz----------- ")
            print(" %4s %40s %12s" %("#","Term","Coeff"))
            for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )
            break
        
        print(" Add operator %4i" %next_index)
        parameters.insert(0,0)
        ansatz_ops.insert(0,pool.fermi_ops[next_index])
        ansatz_mat.insert(0,pool.spmat_ops[next_index])
        
        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)
        

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                options = min_options, method = 'BFGS', callback=trial_model.callback)
    
        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" -----------New ansatz----------- ")
        print(" %4s %40s %12s" %("#","Term","Coeff"))
        for si in range(len(ansatz_ops)):
            s = ansatz_ops[si]
            opstring = ""
            for t in s.terms:
                opstring += str(t)
                break
            print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )

    return
# }}}

def test_lexical(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 1,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-3,
        theta_thresh    = 1e-7,
        adapt_maxiter   = 200,
        pool            = operator_pools.singlet_GSD(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{

    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()
    reference_bra = reference_ket.transpose().conj()

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)

    #Thetas
    parameters = []

    pool.generate_SparseMatrix()
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    
    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []
    curr_state = 1.0*reference_ket

    print(" Now start to grow the ansatz")
    for n_iter in range(0,adapt_maxiter):
    
        print("\n\n\n")
        print(" --------------------------------------------------------------------------")
        print("                         ADAPT-VQE iteration: ", n_iter)                 
        print(" --------------------------------------------------------------------------")
        next_index = None
        next_deriv = 0
        curr_norm = 0
        
        print(" Check each new operator for coupling")
        next_term = []
        print(" Measure commutators:")
        sig = hamiltonian.dot(curr_state)
        for op_trial in range(pool.n_ops):
            
            opA = pool.spmat_ops[op_trial]
            com = 2*(curr_state.transpose().conj().dot(opA.dot(sig))).real
            assert(com.shape == (1,1))
            com = com[0,0]
            assert(np.isclose(com.imag,0))
            com = com.real
            opstring = ""
            for t in pool.fermi_ops[op_trial].terms:
                opstring += str(t)
                break
       
            if abs(com) > adapt_thresh:
                print(" %4i %40s %12.8f" %(op_trial, opstring, com) )

            curr_norm += com*com
            if abs(com) > abs(next_deriv):
                next_deriv = com
                next_index = op_trial

       
        next_index = n_iter % pool.n_ops
        curr_norm = np.sqrt(curr_norm)

        min_options = {'gtol': theta_thresh, 'disp':False}
     
        max_of_com = next_deriv
        print(" Norm of <[A,H]> = %12.8f" %curr_norm)
        print(" Max  of <[A,H]> = %12.8f" %max_of_com)

        converged = False
        if adapt_conver == "norm":
            if curr_norm < adapt_thresh:
                converged = True
        else:
            print(" FAIL: Convergence criterion not defined")
            exit()

        if converged:
            print(" Ansatz Growth Converged!")
            print(" Number of operators in ansatz: ", len(ansatz_ops))
            print(" *Finished: %20.12f" % trial_model.curr_energy)
            print(" -----------Final ansatz----------- ")
            print(" %4s %40s %12s" %("#","Term","Coeff"))
            for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )
            break
        
        print(" Add operator %4i" %next_index)
        parameters.insert(0,0)
        ansatz_ops.insert(0,pool.fermi_ops[next_index])
        ansatz_mat.insert(0,pool.spmat_ops[next_index])
        
        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)
        

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                options = {'gtol': 1e-6, 'disp':True}, method = 'BFGS', callback=trial_model.callback)
    
        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" -----------New ansatz----------- ")
        print(" %4s %40s %12s" %("#","Term","Coeff"))
        for si in range(len(ansatz_ops)):
            s = ansatz_ops[si]
            opstring = ""
            for t in s.terms:
                opstring += str(t)
                break
            print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )

    return
# }}}


def oo_adapt(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 1,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-3,
        theta_thresh    = 1e-7,
        adapt_maxiter   = 200,
        pool            = operator_pools.singlet_GSD(),
        reference       = 'rhf',
        brueckner       = 0,
        ref_state       = None,
        spin_adapt      = True,
        orb_opt         = True,
        include_t1      = True,
        discard_t1_0    = False,
        trotter_t1      = True,
        interleave_t1   = False,
        random_t1       = False,
        t1_order        = None,
        trotter_order   = 1,
        hf_stability    = 'none',
        optimizer       = 'BFGS',
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{

    psi4_time = time.time()
    start_time = time.time()

    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity, reference = reference, hf_stability = hf_stability)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=0, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_bccd = brueckner, # Brueckner CCD
                run_fci=0, 
                delete_input=1)
    pool.init(molecule)
    print(" Psi4 calculation finished in {:<6.3f} seconds".format(time.time() - psi4_time)) 
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    #print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    if brueckner == 1:
        print(' BCCD energy     %20.16f au' %(molecule.bccd_energy))
    if reference == 'rhf':
        print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    jw_time = time.time()
    #Build p-h reference and map it to JW transform
    if ref_state == None:
        ref_state = list(range(0,molecule.n_electrons))

    reference_ket = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(ref_state, molecule.n_qubits)).transpose()
    reference_bra = reference_ket.transpose().conj()

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
    print(" JW mapping finished in {:6.3f} seconds".format(time.time() - jw_time)) 

    pool.generate_SparseMatrix()
    pool.gradient_print_thresh = theta_thresh

    if orb_opt: 
        print(' \nOrbital optimization:\n')

        orb_opt_time = time.time()
        n_t1 = pool.n_t1
        t1_ops = [x/trotter_order for x in pool.fermi_ops[:pool.n_t1]]
        t1_mat = [x/trotter_order for x in pool.spmat_ops[:pool.n_t1]]

        if random_t1:
            t1_order = [i for i in range(n_t1)]
            random.shuffle(t1_order)
            print(' \tRandom ordering of T1 operators')
            print(' \tNew T1 order: %s' % t1_order)
            t1_ops = [pool.fermi_ops[i]/trotter_order for i in t1_order]
            t1_mat = [pool.spmat_ops[i]/trotter_order for i in t1_order]

        elif t1_order != None and random_t1 == False:
            print(' \tUser provided ordering of T1 operators')
            print(' \tNew T1 order: %s' % t1_order)
            t1_ops = [pool.fermi_ops[i]/trotter_order for i in t1_order]
            t1_mat = [pool.spmat_ops[i]/trotter_order for i in t1_order]

        else:
            t1_ops = [x/trotter_order for x in pool.fermi_ops[:pool.n_t1]]
            t1_mat = [x/trotter_order for x in pool.spmat_ops[:pool.n_t1]]

        parameters = [0]*pool.n_t1*trotter_order
        ansatz_ops = t1_ops.copy()
        ansatz_mat = t1_mat.copy()

        for t in range(trotter_order-1):
            ansatz_ops.extend(t1_ops.copy())
            ansatz_mat.extend(t1_mat.copy())

        print(" \tNumber of T1 operators: %s" % pool.n_t1)
        if trotter_t1:
            print(" \tOrbital optimization with Trotterized e^T_1")
            print(" \tTrotter-Suzuki order: %s" % trotter_order)
            trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters, trotter_order)
            curr_state = trial_model.prepare_state(parameters)

        else:
            print(" \tOrbital optimization with non-Trotterized e^T_1")
            print(" \tWarning: this should be taken as NYI.")
            nTS = UCC(hamiltonian, ansatz_mat, reference_ket, parameters, 1)
            curr_state = nTS.prepare_state(parameters)

        print(" \tSetting orbital optimization finished in {:<6.3f} seconds\n".format(time.time() - orb_opt_time)) 

        if include_t1:
            t1_pool = 0
            print(" All T1 and T2 operators in the pool")
        else:
            t1_pool = pool.n_t1
            print(" Only T2 operators in the pool")

        

    else:
        t1_pool = 0
        parameters = []
        ansatz_ops = []     #SQ operator strings in the ansatz
        ansatz_mat = []     #Sparse Matrices for operators in ansatz
        curr_state = 1.0*reference_ket

    print(" Start ADAPT-VQE algorithm")
    print(" Now start to grow the ansatz")
    for n_iter in range(0,adapt_maxiter):
    
        iter_time = time.time()
    
        print("\n\n\n")
        print(" --------------------------------------------------------------------------")
        print("                         ADAPT-VQE iteration: ", n_iter)                 
        print(" --------------------------------------------------------------------------")
        next_index = None
        next_deriv = 0
        curr_norm = 0
        
        print(" Check each new operator for coupling")
        next_term = []
        print(" Measure Operator Pool Gradients:")
        sig = hamiltonian.dot(curr_state)
        e_curr = curr_state.T.conj().dot(sig)[0,0]
        var = sig.T.conj().dot(sig)[0,0] - e_curr**2
        uncertainty = np.sqrt(var.real)
        assert(np.isclose(var.imag,0))
        print(" Variance:    %12.8f" %var.real)
        print(" Uncertainty: %12.8f" %uncertainty)

        for oi in range(t1_pool, pool.n_ops):
            
            gi = pool.compute_gradient_i(oi, curr_state, sig)

            curr_norm += gi*gi
            if abs(gi) > abs(next_deriv):
                next_deriv = gi
                next_index = oi

        curr_norm = np.sqrt(curr_norm)

        min_options = {'gtol': theta_thresh, 'disp':False}
     
        max_of_gi = next_deriv
        print(" Norm of <[H,A]> = %12.8f" %curr_norm)
        print(" Max  of <[H,A]> = %12.8f" %max_of_gi)

        converged = False
        if adapt_conver == "norm":
            if curr_norm < adapt_thresh:
                converged = True
        elif adapt_conver == "var":
            if abs(var) < adapt_thresh:
                #variance
                converged = True
        else:
            print(" FAIL: Convergence criterion not defined")
            exit()

        if converged:
            print(" Ansatz Growth Converged!")
            print(" Number of operators in ansatz: ", len(ansatz_ops))
            print(" *Finished: %20.12f" % trial_model.curr_energy)
            print(" -----------Final ansatz----------- ")
            print(" %4s %12s %18s" %("#","Coeff","Term"))
            for si in range(len(ansatz_ops)):
                opstring = pool.get_string_for_term(ansatz_ops[si])
                print(" %4i %12.8f %s" %(si, parameters[si], opstring) )
            build_circuit(molecule.n_electrons, molecule.n_qubits, ansatz_ops, parameters[:len(ansatz_ops)])
            print("\n ADAPT-VQE simulation finished in {:<6.3f} seconds".format(time.time() - start_time)) 
            break
        
        if interleave_t1 and n_iter !=0:
            parameters = [0]*n_t1 + parameters
            ansatz_ops = t1_ops.copy() + ansatz_ops
            ansatz_mat = t1_mat.copy() + ansatz_mat

        print(" Add operator %4i" %next_index)
        parameters.insert(0,0)
        ansatz_ops.insert(0,pool.fermi_ops[next_index])
        ansatz_mat.insert(0,pool.spmat_ops[next_index])

        trial_model = tUCCSD(hamiltonian, ansatz_mat, curr_state, parameters, trotter_order)

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                options = min_options, method = optimizer, callback=trial_model.callback)
    
        parameters = list(opt_result['x'])

        curr_state = trial_model.prepare_state(parameters)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" Iteration finished in {:<6.3f} seconds".format(time.time() - iter_time)) 
        print(" -----------New ansatz----------- ")
        print(" %4s %12s %18s" %("#","Coeff","Term"))
        for si in range(len(ansatz_ops)):
            opstring = pool.get_string_for_term(ansatz_ops[si])
            print(" %4i %12.8f %s" %(si, parameters[si], opstring) )

        if discard_t1_0 and orb_opt and n_iter == 0:
            new_params = []
            new_ops = []
            new_mat = []

            if interleave_t1:
                for si in range(pool.n_t1, 0, -1):
                    if abs(parameters[si]) >= 1e-12:
                        new_params.append(parameters[si])
                        new_ops.append(ansatz_ops[si])
                        new_mat.append(ansatz_mat[si])

                #parameters = new_params
                #ansatz_ops = new_ops
                #ansatz_mat = new_mat
                n_t1 = len(new_params)
                t1_ops = new_ops.copy()
                t1_mat = new_mat.copy()

                new_params.insert(0,parameters[0])
                new_ops.insert(0,ansatz_ops[0])
                new_mat.insert(0,ansatz_mat[0])
                
                parameters = new_params
                ansatz_ops = new_ops
                ansatz_mat = new_mat

            else:
                for si in range(len(parameters)):
                    if abs(parameters[si]) >= 1e-12:
                        new_params.append(parameters[si])
                        new_ops.append(ansatz_ops[si])
                        new_mat.append(ansatz_mat[si])

                parameters = new_params
                ansatz_ops = new_ops
                ansatz_mat = new_mat
            
            if trotter_t1:
                trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters, trotter_order)
                curr_state = trial_model.prepare_state(parameters)

            else:
                nTS = UCC(hamiltonian, ansatz_mat, reference_ket, parameters, 1)
                curr_state = nTS.prepare_state(parameters)
# }}}

def build_circuit(n_electrons, n_qubits, state, opt_amplitudes):

    print("\n\n")
    print(" --------------------------------------------------------------------------")
    print("                Circuit for optimized ADAPT-VQE ansatz                     ")                 
    print(" --------------------------------------------------------------------------")
    compiler_engine = uccsd_trotter_engine()
    compiler_engine = uccsd_trotter_engine(CommandPrinter())
    wavefunction = compiler_engine.allocate_qureg(n_qubits)
    for i in range(n_electrons):
        X | wavefunction[i]

    # generator store all the fermionic operators in the ADAPT-VQE ansatz
    generator = FermionOperator()
    for i in state:
        generator += i

    evolution_operator = uccsd_singlet_evolution(opt_amplitudes, n_qubits, n_electrons, opt_state = generator)
    evolution_operator | wavefunction
    compiler_engine.flush()

def exclude_t1(parameters):

    idx = []
    print(parameters) 
    for i in range(len(parameters)):
        if abs(parameters[i]) >= 1e-12:
            idx.append(si)
    return idx

if __name__== "__main__":
    r = 1.5
    #geometry = [('H', (0,0,1*r)), ('H', (0,0,2*r)), ('H', (0,0,3*r)), ('H', (0,0,4*r))]
    geometry = [('H',  (0, 0, 0)), 
                ('Li', (0, 0, r*2.39))]
    #geometry = [('H', (0,0,1*r)), ('H', (0,0,2*r)), ('H', (0,0,3*r)), ('H', (0,0,4*r)), ('H', (0,0,5*r)), ('H', (0,0,6*r))]

    #vqe_methods.ucc(geometry,pool = operator_pools.singlet_SD())
    #vqe_methods.adapt_vqe(geometry,pool = operator_pools.singlet_SD())
    #vqe_methods.adapt_vqe(geometry,pool = operator_pools.hamiltonian(), adapt_thresh=1e-7, theta_thresh=1e-8)
    #vqe_methods.adapt_vqe(geometry,pool = operator_pools.singlet_SD(), adapt_thresh=1e-1, adapt_conver='uncertainty')
    vqe_methods.adapt_vqe(geometry,pool = operator_pools.spin_complement_GSD(), adapt_thresh=1e-2, theta_thresh=1e-9)

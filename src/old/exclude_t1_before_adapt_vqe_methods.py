import scipy
import openfermion
import openfermionpsi4
import os
import numpy as np
import copy
import random 
import sys

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
        k               = 1, # # of products of unitaries in k-UpCCGSD
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
    #parameters = [0]*pool.n_ops 
    parameters = np.random.rand(2*pool.n_ops)
    print(" Initial parameters: %s" % parameters)
    pool.generate_SparseMatrix()
    #print(pool.spmat_ops)
    
    #ucc = tUCCSD(hamiltonian, pool.spmat_ops, reference_ket, parameters, k)
    ucc = UCC(hamiltonian, pool.spmat_ops, reference_ket, parameters, k)
    print(ucc)
    
    opt_result = scipy.optimize.minimize(ucc.energy, 
                parameters, options = {'gtol': 1e-6, 'disp':True}, 
                method = 'BFGS', callback=ucc.callback)
    print(" Finished: %20.12f" % ucc.curr_energy)
    parameters = opt_result['x']
    for p in parameters:
        print(p)


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
        include_t1      = True,
        trotter_t1      = True,
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

    #Build p-h reference and map it to JW transform
    if ref_state == None:
        ref_state = list(range(0,molecule.n_electrons))

    reference_ket = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(ref_state, molecule.n_qubits)).transpose()
    reference_bra = reference_ket.transpose().conj()

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)

    #reference_ket = uccs.prepare_state(parameters)

    pool.generate_SparseMatrix()
    pool.gradient_print_thresh = theta_thresh
    
    ansatz_ops = pool.fermi_ops[:pool.n_t1]     #SQ operator strings in the ansatz
    ansatz_mat = pool.spmat_ops[:pool.n_t1]     #Sparse Matrices for operators in ansatz
    
    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = [0]*pool.n_t1
    #curr_state = 1.0*reference_ket

    print(" Number of T1 operators: %s" % pool.n_t1)
    if trotter_t1:
        print(" Orbital optimization with Trotterized e^T_1")
        TS = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)
        #singles_state = TS.prepare_state(parameters)
        #opt_result = scipy.optimize.minimize(TS.energy, parameters, jac=TS.gradient, 
        #        options = {'gtol': 1e-6, 'disp':True}, method = 'BFGS', callback=TS.callback)
        #parameters = list(opt_result['x'])
        #print(parameters)
        #idx = exclude_t1(parameters)
        #parameters = [parameters[i] for i in idx]
        #ansatz_ops = [ansatz_ops[i] for i in idx]
        #ansatz_mat = [ansatz_mat[i] for i in idx]
        #TS = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)
        curr_state = TS.prepare_state(parameters)

    else:
        print(" Orbital optimization with non-Trotterized e^T_1")
        nTS = UCC(hamiltonian, ansatz_mat, reference_ket, parameters, 1)
        singles_state = nTS.prepare_state(parameters)
        opt_result = scipy.optimize.minimize(nTS.energy, 
                    parameters, options = {'gtol': 1e-6, 'disp':True}, 
                    method = 'BFGS', callback=nTS.callback)
        print(" Finished non-Trotterized e^T_1 optimization: %20.12f" % nTS.curr_energy)
        parameters = opt_result['x']
        idx = exclude_t1(parameters)
        parameters = [parameters[i] for i in idx]
        ansatz_ops = [ansatz_ops[i] for i in idx]
        ansatz_mat = [ansatz_mat[i] for i in idx]
        nTS = UCC(hamiltonian, ansatz_mat, reference_ket, parameters, 1)
        curr_state = nTS.prepare_state(parameters)


    if include_t1:
        t1_pool = 0
        print(" All T1 and T2 operators in the pool")
    else:
        t1_pool = pool.n_t1
        print(" Only T2 operators in the pool")


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

        for oi in range(t1_pool, pool.n_ops):
        #for oi in range(pool.n_t1, pool.n_ops):
            
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
            build_circuit(molecule.n_electrons, molecule.n_qubits, parameters[:len(ansatz_ops)])
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

        if n_iter == 1000:
            new_params = []
            new_ops = []
            new_mat = []
            for si in range(len(parameters)):
                if abs(parameters[si]) >= 1e-12:
                    print(parameters[si])
                    new_params.append(parameters[si])
                    new_ops.append(ansatz_ops[si])
                    new_mat.append(ansatz_mat[si])

            parameters = new_params
            ansatz_ops = new_ops
            ansatz_mat = new_mat
    
# }}}

def build_circuit(n_electrons, n_qubits, opt_amplitudes):

    print("\n\n")
    print(" --------------------------------------------------------------------------")
    print("                Circuit for optimized ADAPT-VQE ansatz                     ")                 
    print(" --------------------------------------------------------------------------")
    compiler_engine = uccsd_trotter_engine()
    compiler_engine = uccsd_trotter_engine(CommandPrinter())
    wavefunction = compiler_engine.allocate_qureg(n_qubits)
    for i in range(n_electrons):
        X | wavefunction[i]

    # Build the circuit and act it on the wavefunction
    evolution_operator = uccsd_singlet_evolution(opt_amplitudes, n_qubits, n_electrons)
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

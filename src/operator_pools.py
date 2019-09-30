import openfermion
import numpy as np
import copy as cp
import itertools
import scipy

from openfermion import *



class OperatorPool:
    def __init__(self):
        self.n_orb = 0
        self.n_occ_a = 0
        self.n_occ_b = 0
        self.n_vir_a = 0
        self.n_vir_b = 0

        self.n_spin_orb = 0
        self.gradient_print_thresh = 0

    def init(self,molecule):
        self.molecule = molecule
        self.n_orb = molecule.n_orbitals
        self.n_spin_orb = 2*self.n_orb 
        self.n_occ_a = molecule.get_n_alpha_electrons()
        self.n_occ_b = molecule.get_n_beta_electrons()
    
        self.n_vir_a = self.n_orb - self.n_occ_a
        self.n_vir_b = self.n_orb - self.n_occ_b
        
        self.n_occ = self.n_occ_a
        self.n_vir = self.n_vir_a
        self.n_ops = 0

        self.generate_SQ_Operators()

    def generate_SQ_Operators(self):
        print("Virtual: Reimplement")
        exit()

    def generate_SparseMatrix(self):
        self.spmat_ops = []
        print(" Generate Sparse Matrices for operators in pool")
        for op in self.fermi_ops:
            self.spmat_ops.append(transforms.get_sparse_operator(op, n_qubits = self.n_spin_orb))
        assert(len(self.spmat_ops) == self.n_ops)
        return

    def get_string_for_term(self,op):

        opstring = ""
        spins = ""
        for t in op.terms:
       
            
            opstring = "("
            for ti in t:
                opstring += str(int(ti[0]/2))
                if ti[1] == 0:
                    opstring += "  "
                elif ti[1] == 1:
                    opstring += "' "
                else:
                    print("wrong")
                    exit()
                spins += str(ti[0]%2)

#            if self.fermi_ops[i].terms[t] > 0:
#                spins = "+"+spins
#            if self.fermi_ops[i].terms[t] < 0:
#                spins = "-"+spins
            opstring += ")" 
            spins += " "
        opstring = " %18s : %s" %(opstring, spins)
        return opstring

    def overlap_gradient_i(self, op_index, curr_state, fci):

        op = self.spmat_ops[op_index]
        op_v = op.dot(curr_state)

        index = scipy.sparse.find(op_v)[0]
        coeff = scipy.sparse.find(op_v)[2]
        grad = 0.0
        for i in range(len(index)):
            grad += fci[index[i]]*coeff[i]

        grad = np.absolute(grad)
        opstring = self.get_string_for_term(self.fermi_ops[op_index])

        if abs(grad) > self.gradient_print_thresh:
            print(" %4i %12.8f %s" %(op_index, grad, opstring) )
    
        return grad 


    def compute_gradient_i(self,i,v,sig):
        """
        For a previously optimized state |n>, compute the gradient g(k) of exp(c(k) A(k))|n>
        g(k) = 2Real<HA(k)>

        Note - this assumes A(k) is an antihermitian operator. If this is not the case, the derived class should 
        reimplement this function. Of course, also assumes H is hermitian

        v   = current_state
        sig = H*v

        """
        opA = self.spmat_ops[i]
        gi = 2*(sig.transpose().conj().dot(opA.dot(v)))
        assert(gi.shape == (1,1))
        gi = gi[0,0]
        assert(np.isclose(gi.imag,0))
        gi = gi.real
       
        opstring = self.get_string_for_term(self.fermi_ops[i])

        if abs(gi) > self.gradient_print_thresh:
            print(" %4i %12.8f %s" %(i, gi, opstring) )
    
        return gi
   

class spin_complement_GSD(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        alpha_orbs = [2*i for i in range(self.n_orb)]
        beta_orbs = [2*i+1 for i in range(self.n_orb)]
    
        ops = []
        #aa
        for p in alpha_orbs:
            for q in alpha_orbs:
                if p>=q:
                    continue
                #if abs(hamiltonian_op.one_body_tensor[p,q]) < 1e-8:
                #    print(" Dropping term %4i %4i" %(p,q), " V= %+6.1e" %hamiltonian_op.one_body_tensor[p,q])
                #    continue
                one_elec = openfermion.FermionOperator(((q,1),(p,0)))-openfermion.FermionOperator(((p,1),(q,0)))
                one_elec += openfermion.FermionOperator(((q+1,1),(p+1,0)))-openfermion.FermionOperator(((p+1,1),(q+1,0)))
                ops.append(one_elec)
        #aa
        pq = 0
        for p in alpha_orbs:
            for q in alpha_orbs:
                if p>q:
                    continue
                rs = 0
                for r in alpha_orbs:
                    for s in alpha_orbs:
                        if r>s:
                            continue
                        if pq<rs:
                            continue
                        #if abs(hamiltonian_op.two_body_tensor[p,r,s,q]) < 1e-8:
                            #print(" Dropping term %4i %4i %4i %4i" %(p,r,s,q), " V= %+6.1e" %hamiltonian_op.two_body_tensor[p,r,s,q])
                            #continue
                        two_elec = openfermion.FermionOperator(((r,1),(p,0),(s,1),(q,0)))-openfermion.FermionOperator(((q,1),(s,0),(p,1),(r,0)))
                        two_elec += openfermion.FermionOperator(((r+1,1),(p+1,0),(s+1,1),(q+1,0)))-openfermion.FermionOperator(((q+1,1),(s+1,0),(p+1,1),(r+1,0)))
                        ops.append(two_elec)
                        rs += 1
                pq += 1
        
        
        #ab
        pq = 0
        for p in alpha_orbs:
            for q in beta_orbs:
                rs = 0
                for r in alpha_orbs:
                    for s in beta_orbs:
                        if pq<rs:
                            continue
                        two_elec = openfermion.FermionOperator(((r,1),(p,0),(s,1),(q,0)))-openfermion.FermionOperator(((q,1),(s,0),(p,1),(r,0)))
                        if p>q:
                            continue
                        two_elec += openfermion.FermionOperator(((s-1,1),(q-1,0),(r+1,1),(p+1,0)))-openfermion.FermionOperator(((p+1,1),(r+1,0),(q-1,1),(s-1,0)))
                        ops.append(two_elec)
                        rs += 1
                pq += 1

        self.fermi_ops = ops
        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 
# }}}




class spin_complement_GSD2(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        
        print(" Form spin-complemented GSD operators")
        
        self.fermi_ops = []
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                termA =  FermionOperator(((pa,1),(qa,0)))
                termA += FermionOperator(((pb,1),(qb,0)))
 
                termA -= hermitian_conjugated(termA)
               
                termA = normal_ordered(termA)
                
                if termA.many_body_order() > 0:
                    self.fermi_ops.append(termA)
                       
      
        pq = -1 
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                pq += 1
        
                rs = -1 
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1
                    
                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1
                    
                        rs += 1
                    
                        if(pq > rs):
                            continue
                        
                        termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)))
                        termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)))
                        
                        termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)))
                        termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)))
                        
                        termC =  FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)))
                        termC += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)))

#                        termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)))
#                        termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)))
#                                                                      
#                        termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)))
#                        termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)))
#                        
#                        termC =  FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)))
#                        termC += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)))

#                        print()
#                        print(p,q,r,s)
#                        print(termA)
#                        print(termB)
#                        print(termC)
                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)
                        termC -= hermitian_conjugated(termC)
               
                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)
                        termC = normal_ordered(termC)
                        
                        if termA.many_body_order() > 0:
                            self.fermi_ops.append(termA)
                        
                        if termB.many_body_order() > 0:
                            self.fermi_ops.append(termB)

                        if termC.many_body_order() > 0:
                            self.fermi_ops.append(termC)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 
# }}}


class singlet_GSD(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        
        print(" Form singlet GSD operators")
        
        self.fermi_ops = []
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                termA =  FermionOperator(((pa,1),(qa,0)))
                termA += FermionOperator(((pb,1),(qb,0)))
 
                termA -= hermitian_conjugated(termA)
               
                termA = normal_ordered(termA)
                
                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t
            
                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)

        self.n_t1 = len(self.fermi_ops)
                       
        pq = -1 
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                pq += 1
        
                rs = -1 
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1
                    
                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1
                    
                        rs += 1
                    
                        if(pq > rs):
                            continue

                        termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))
                                                                      
                        termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)),  1/2.0)
                        termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)),  1/2.0)
                        termB += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), -1/2.0)
                        termB += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), -1/2.0)
 
                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)
               
                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)
                        
                        #Normalize
                        coeffA = 0
                        coeffB = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t

                        
                        if termA.many_body_order() > 0:
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)
                        
                        if termB.many_body_order() > 0:
                            termB = termB/np.sqrt(coeffB)
                            self.fermi_ops.append(termB)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 
# }}}




class singlet_SD(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        """
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form singlet SD operators")
        self.fermi_ops = [] 
       
        n_occ = self.n_occ
        n_vir = self.n_vir
       
        for i in range(0,n_occ):
            ia = 2*i
            ib = 2*i+1

            for a in range(0,n_vir):
                aa = 2*n_occ + 2*a
                ab = 2*n_occ + 2*a+1
                    
                termA =  FermionOperator(((aa,1),(ia,0)), 1/np.sqrt(2))
                termA += FermionOperator(((ab,1),(ib,0)), 1/np.sqrt(2))
                
                termA -= hermitian_conjugated(termA)
                        
                termA = normal_ordered(termA)
               
                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t
                
                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)
       
        self.n_t1 = len(self.fermi_ops)

        for i in range(0,n_occ):
            ia = 2*i
            ib = 2*i+1

            for j in range(i,n_occ):
                ja = 2*j
                jb = 2*j+1
        
                for a in range(0,n_vir):
                    aa = 2*n_occ + 2*a
                    ab = 2*n_occ + 2*a+1

                    for b in range(a,n_vir):
                        ba = 2*n_occ + 2*b
                        bb = 2*n_occ + 2*b+1

                        termA =  FermionOperator(((aa,1),(ba,1),(ia,0),(ja,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(bb,1),(ib,0),(jb,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((aa,1),(bb,1),(ia,0),(jb,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(ba,1),(ib,0),(ja,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(ba,1),(ia,0),(jb,0)), 1/np.sqrt(12))
                                                                      
                        termB  = FermionOperator(((aa,1),(bb,1),(ia,0),(jb,0)), 1/2)
                        termB += FermionOperator(((ab,1),(ba,1),(ib,0),(ja,0)), 1/2)
                        termB += FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), -1/2)
                        termB += FermionOperator(((ab,1),(ba,1),(ia,0),(jb,0)), -1/2)
                
                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)
               
                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)
                        
                        #Normalize
                        coeffA = 0
                        coeffB = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t

                        
                        if termA.many_body_order() > 0:
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)
                        
                        if termB.many_body_order() > 0:
                            termB = termB/np.sqrt(coeffB)
                            self.fermi_ops.append(termB)
        
        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 
    # }}}


#def unrestricted_SD(n_occ_a, n_occ_b, n_vir_a, n_vir_b):
    #print("NYI")
    #exit()

class sf_rohf_SD(OperatorPool):
    def generate_SQ_Operators(self):
        """
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form restricted open-shell SD operators")
        self.fermi_ops = []  
    
        n_occ_a = self.n_occ_a
        n_occ_b = self.n_occ_b
        n_vir_a = self.n_vir_a
        n_vir_b = self.n_vir_b
        n_orb   = self.n_orb

        print('Alpha: %s' % n_vir_a)
        print('Beta: %s' % n_vir_b)
        # Normal ROHF excitations

        for spin in ['a','b']:
            if spin == 'a':
                n_occ = n_occ_a
                n_vir = n_vir_a
                n_occ_shift = 0
                n_vir_shift = 2*n_occ_a
            else:
                n_occ = n_occ_b
                n_vir = n_vir_b
                n_occ_shift = 1
                n_vir_shift = 2*n_occ_b + 1

            for i in range(0,n_occ):
                ii = 2*i + n_occ_shift # ii = i index 

                for a in range(0,n_vir):
                    ai = 2*a + n_vir_shift # ai = a index 

                    termA =  FermionOperator(((ai,1),(ii,0)), 1.0)
                    termA -= hermitian_conjugated(termA)
                    termA = normal_ordered(termA)
                    #Normalize

                    coeffA = 0 
                    for t in termA.terms:
                        coeff_t = termA.terms[t]
                        coeffA += coeff_t * coeff_t
        
                    if termA.many_body_order() > 0:
                        termA = termA/np.sqrt(coeffA)
                        self.fermi_ops.append(termA)    

        # Spin flips
        for i in range(0, n_occ_a - n_occ_b):
            ia = 2*n_occ_b + 2*i

            for a in range(0, n_occ_a - n_occ_b):
                ab = 2*n_occ_b + 2*a + 1

                termA =  FermionOperator(((ab,1),(ia,0)), 1.0)
                termA -= hermitian_conjugated(termA)
                termA = normal_ordered(termA)
                print(termA)

                #Normalize

                coeffA = 0 
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t
    
                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)    

# same spin

        for spin in ['a','b']:
            if spin == 'a':
                n_occ = n_occ_a
                n_vir = n_vir_a
                n_occ_shift = 0
                n_vir_shift = 2*n_occ_a
            else:
                n_occ = n_occ_b
                n_vir = n_vir_b
                n_occ_shift = 1
                n_vir_shift = 2*n_occ_b + 1

            for i in range(0,n_occ):
                ii = 2*i + n_occ_shift 

                for j in range(i+1,n_occ):
                    jj = 2*j + n_occ_shift 
                    
                    for a in range(0,n_vir_a):
                        aa = 2*a + n_vir_shift

                        for b in range(a+1,n_vir_a):
                            bb = 2*b + n_vir_shift

                            termA =  FermionOperator(((aa,1),(bb,1),(ii,0),(jj,0)), 1.0)
                            print(termA)
                            termA -= hermitian_conjugated(termA)
                            termA = normal_ordered(termA)

                            coeffA = 0
                            for t in termA.terms:
                                coeff_t = termA.terms[t]
                                coeffA += coeff_t * coeff_t
                            
                            if termA.many_body_order() > 0:
                                termA = termA/np.sqrt(coeffA)
                                self.fermi_ops.append(termA)

# Opposite spins 

        for i in range(0, n_occ_a - n_occ_b):
            ia = 2*i

            for j in range(0, n_occ_a - n_occ_b):
                ja = 2*j

                for a in range(0, n_occ_a - n_occ_b):
                    ab = 2*a + 1 

                    for b in range(0, n_occ_a - n_occ_b):
                        bb = 2*b + 1

                        termA =  FermionOperator(((ab,1),(ia,0),(bb,1),(ja,0)), 1.0)
                        print(termA)
                        termA -= hermitian_conjugated(termA)
                        termA = normal_ordered(termA)
                                                                      
                        #Normalize
                        coeffA = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        
                        if termA.many_body_order() > 0:
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 

class unrestricted_SD(OperatorPool):
    def generate_SQ_Operators(self):
        """
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form unrestricted SD operators")
        self.fermi_ops = []  
    
        n_occ_a = self.n_occ_a
        n_occ_b = self.n_occ_b
        n_vir_a = self.n_vir_a
        n_vir_b = self.n_vir_b


        for spin in ['a','b']:
            if spin == 'a':
                n_occ = n_occ_a
                n_vir = n_vir_a
                n_occ_shift = 0 
                n_vir_shift = 2*n_occ_a
            else:
                n_occ = n_occ_b
                n_vir = n_vir_b
                n_occ_shift = 1 
                n_vir_shift = 2*n_occ_b + 1 

            for i in range(0,n_occ):
                ii = 2*i + n_occ_shift # ii = i index 

                for a in range(0,n_vir):
                    ai = 2*a + n_vir_shift # ai = a index 

                    termA =  FermionOperator(((ai,1),(ii,0)), 1.0)
                    termA -= hermitian_conjugated(termA)
                    termA = normal_ordered(termA)

                    #Normalize
                    coeffA = 0 
                    for t in termA.terms:
                        coeff_t = termA.terms[t]
                        coeffA += coeff_t * coeff_t
    
                    if termA.many_body_order() > 0:
                        termA = termA/np.sqrt(coeffA)
                        self.fermi_ops.append(termA)    


        for spin_1 in ['a','b']:
            if spin_1 == 'a':
                n_occ_1 = n_occ_a
                n_vir_1 = n_vir_a
                n_occ_shift_1 = 0 
                n_vir_shift_1 = 2*n_occ_a
            else:
                n_occ_1 = n_occ_b
                n_vir_1 = n_vir_b
                n_occ_shift_1 = 1 
                n_vir_shift_1 = 2*n_occ_b + 1 

#  
            for i in range(0,n_occ_1):
                ii = 2*i + n_occ_shift_1 # ii = i index 
    
                for spin_2 in ['a','b']:
                    if spin_2 == 'a':
                        n_occ_2 = n_occ_a
                        n_vir_2 = n_vir_a
                        n_occ_shift_2 = 0 
                        n_vir_shift_2 = 2*n_occ_a
                    else:
                        n_occ_2 = n_occ_b
                        n_vir_2 = n_vir_b
                        n_occ_shift_2 = 1 
                        n_vir_shift_2 = 2*n_occ_b + 1 

                    for j in range(0,n_occ_2):
                        ji = 2*j + n_occ_shift_2
    
                        for a in range(0,n_vir_1):
                            ai = 2*a + n_vir_shift_1

                            for b in range(0,n_vir_2):
                                bi = 2*b + n_vir_shift_2
    
                                termA =  FermionOperator(((ai,1),(bi,1),(ii,0),(ji,0)), 1.0)
                                termA -= hermitian_conjugated(termA)
                                termA = normal_ordered(termA)

                                #Normalize
                                coeffA = 0 
                                for t in termA.terms:
                                    coeff_t = termA.terms[t]
                                    coeffA += coeff_t * coeff_t

                                if termA.many_body_order() > 0:
                                    termA = termA/np.sqrt(coeffA)
                                    self.fermi_ops.append(termA)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return

class vacuum(OperatorPool):
    def generate_SQ_Operators(self):
        """
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form vacuum operators")
        self.fermi_ops = []  
   
        n_electrons = self.n_orb
        n_orb = 2*self.n_orb 
       
        x = QubitOperator('X0')
        print(x)
        self.fermi_ops.append(x)
        single_ops = []
        for i in range(n_orb):

            for dagger in range(0,2):

                string = (i,dagger)
                single_ops.append(string)
                termA =  FermionOperator((string), 1.0)
                termA -= hermitian_conjugated(termA)
                print(termA)
                
                termA = normal_ordered(termA)
                #Normalize

                coeffA = 0 
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)    

        #ops = single_ops
        for i in range(1, n_orb):

            ops = itertools.permutations(single_ops, i+1)
            #ops = itertools.combinations(single_ops, i+1)

            for string in ops:

                termA =  FermionOperator(string, 1.0)
                if termA != hermitian_conjugated(termA) and str(termA).count('^') >= (i+1)/2:
                
                    termA -= hermitian_conjugated(termA)
                    termA = normal_ordered(termA)

                    coeffA = 0 
                    for t in termA.terms:
                        coeff_t = termA.terms[t]
                        coeffA += coeff_t * coeff_t

                    if termA.many_body_order() > 0:
                        termA = termA/np.sqrt(coeffA)
                        self.fermi_ops.append(termA)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 

class sf_hs_rohf_GSD(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        
        print(" Form spin-flip high-spin ROHF operators")
        
        self.fermi_ops = []
        for p in range(0,2*self.n_orb):
 
            for q in range(p,2*self.n_orb):
        
                termA =  FermionOperator(((p,1),(q,0)))
                termA -= hermitian_conjugated(termA)
                termA = normal_ordered(termA)
                
                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t
            
                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)

        pq = -1 
        for p in range(0,2*self.n_orb):
 
            for q in range(p,2*self.n_orb):
        
                pq += 1
        
                rs = -1 
                for r in range(0,2*self.n_orb):
                    
                    for s in range(r,2*self.n_orb):
                    
                        rs += 1
                    
                        if(pq > rs):
                            continue

                        termA =  FermionOperator(((r,1),(p,0),(s,1),(q,0)),1.0 )
                        termA -= hermitian_conjugated(termA)
                        termA = normal_ordered(termA)
                        
                        #Normalize
                        coeffA = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        
                        if termA.many_body_order() > 0:
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 

class UpCCGSD(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        
        print(" Form unitary pair CC GSD operators")
        
        self.fermi_ops = []
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p + 1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q + 1
        
                termA =  FermionOperator(((pa,1),(qa,0)))
                termA += FermionOperator(((pb,1),(qb,0)))
                termA -= hermitian_conjugated(termA)
                termA = normal_ordered(termA)
                
                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t
            
                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)

                termA =  FermionOperator(((pa,1),(pb,1),(qb,0),(qa,0)), 1.0)
                termA -= hermitian_conjugated(termA)
                termA = normal_ordered(termA)
                
                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t
                
                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 
        # }}}

class unrestricted_GSD(OperatorPool):
    def generate_SQ_Operators(self):
        """
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form unrestricted GSD operators")
        self.fermi_ops = []  
    
        # 0 is alpha and 1 is beta
        for shift in [0,1]:

            for p in range(0,self.n_orb):
                pp = 2*p + shift # ii = i index 

                for q in range(p,self.n_orb):
                    qq = 2*q + shift # ai = a index 

                    termA =  FermionOperator(((qq,1),(pp,0)), 1.0)
                    termA -= hermitian_conjugated(termA)
                    termA = normal_ordered(termA)

                    #Normalize
                    coeffA = 0 
                    for t in termA.terms:
                        coeff_t = termA.terms[t]
                        coeffA += coeff_t * coeff_t
    
                    if termA.many_body_order() > 0:
                        termA = termA/np.sqrt(coeffA)
                        self.fermi_ops.append(termA)    

        self.n_t1 = len(self.fermi_ops)

        for shift_1 in [0,1]:

            for p in range(0,self.n_orb):
                pp = 2*p + shift_1 # ii = i index 
    
                for shift_2 in [0,1]:

                    for q in range(p, self.n_orb):
                        qq = 2*q + shift_2
    
                        for r in range(0,self.n_orb):
                            rr = 2*r + shift_1

                            for s in range(r,self.n_orb):
                                ss = 2*s + shift_2
    
                                termA =  FermionOperator(((rr,1),(pp,0),(ss,1),(qq,0)), 1.0)
                                termA -= hermitian_conjugated(termA)
                                termA = normal_ordered(termA)

                                #Normalize
                                coeffA = 0 
                                for t in termA.terms:
                                    coeff_t = termA.terms[t]
                                    coeffA += coeff_t * coeff_t

                                if termA.many_body_order() > 0:
                                    termA = termA/np.sqrt(coeffA)
                                    self.fermi_ops.append(termA)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 

class singlet_BGD(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        
        print(" Form singlet Brueckner GD operators")
        
        self.fermi_ops = []
                       
        pq = -1 
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                pq += 1
        
                rs = -1 
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1
                    
                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1
                    
                        rs += 1
                    
                        if(pq > rs):
                            continue

                        termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))
                                                                      
                        termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)),  1/2.0)
                        termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)),  1/2.0)
                        termB += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), -1/2.0)
                        termB += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), -1/2.0)
 
                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)
               
                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)
                        
                        #Normalize
                        coeffA = 0
                        coeffB = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t

                        
                        if termA.many_body_order() > 0:
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)
                        
                        if termB.many_body_order() > 0:
                            termB = termB/np.sqrt(coeffB)
                            self.fermi_ops.append(termB)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 

class singlet_BD(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        """
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form normal Brueckner double operators")
        self.fermi_ops = [] 
       
        n_occ = self.n_occ
        n_vir = self.n_vir
       
        for i in range(0,n_occ):
            ia = 2*i
            ib = 2*i+1

            for j in range(i,n_occ):
                ja = 2*j
                jb = 2*j+1
        
                for a in range(0,n_vir):
                    aa = 2*n_occ + 2*a
                    ab = 2*n_occ + 2*a+1

                    for b in range(a,n_vir):
                        ba = 2*n_occ + 2*b
                        bb = 2*n_occ + 2*b+1

                        termA =  FermionOperator(((aa,1),(ba,1),(ia,0),(ja,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(bb,1),(ib,0),(jb,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((aa,1),(bb,1),(ia,0),(jb,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(ba,1),(ib,0),(ja,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(ba,1),(ia,0),(jb,0)), 1/np.sqrt(12))
                                                                      
                        termB  = FermionOperator(((aa,1),(bb,1),(ia,0),(jb,0)), 1/2)
                        termB += FermionOperator(((ab,1),(ba,1),(ib,0),(ja,0)), 1/2)
                        termB += FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), -1/2)
                        termB += FermionOperator(((ab,1),(ba,1),(ia,0),(jb,0)), -1/2)
                
                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)
               
                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)
                        
                        #Normalize
                        coeffA = 0
                        coeffB = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t

                        
                        if termA.many_body_order() > 0:
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)
                        
                        if termB.many_body_order() > 0:
                            termB = termB/np.sqrt(coeffB)
                            self.fermi_ops.append(termB)
        
        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 
    # }}}

class singlet_GS(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        
        print(" Form singlet generalized singles operators")
        
        self.fermi_ops = []
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                termA =  FermionOperator(((pa,1),(qa,0)))
                termA += FermionOperator(((pb,1),(qb,0)))
 
                termA -= hermitian_conjugated(termA)
               
                termA = normal_ordered(termA)
                
                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t
            
                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)
                       
        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 

class oo_singlet_GSD(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        
        print(" Form singlet GSD operators (same as GSD)")
        
        self.fermi_ops = []
        self.generator = FermionOperator()
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                termA =  FermionOperator(((pa,1),(qa,0)))
                termA += FermionOperator(((pb,1),(qb,0)))
 
                termA -= hermitian_conjugated(termA)
               
                termA = normal_ordered(termA)
                self.generator += termA
                
                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t
            
                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)

        self.n_t1 = len(self.fermi_ops)

        pq = -1 
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                pq += 1
        
                rs = -1 
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1
                    
                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1
                    
                        rs += 1
                    
                        if(pq > rs):
                            continue

                        termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))
                                                                      
                        termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)),  1/2.0)
                        termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)),  1/2.0)
                        termB += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), -1/2.0)
                        termB += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), -1/2.0)
 
                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)
                        self.generator += termA
               
                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)
                        self.generator += termB
                        
                        #Normalize
                        coeffA = 0
                        coeffB = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t

                        
                        if termA.many_body_order() > 0:
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)
                        
                        if termB.many_body_order() > 0:
                            termB = termB/np.sqrt(coeffB)
                            self.fermi_ops.append(termB)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 
# }}}

class GSD_singles_notspinadapt(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        
        print(" Form GSD operators with non-spin adapted singles")
        
        self.fermi_ops = []
        for p in range(0,2*self.n_orb):
 
            for q in range(p,2*self.n_orb):
        
                termA =  FermionOperator(((p,1),(q,0)))
                print(termA)
                termA -= hermitian_conjugated(termA)
                termA = normal_ordered(termA)
                
                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t
            
                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)

        self.n_t1 = len(self.fermi_ops)

        pq = -1 
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                pq += 1
        
                rs = -1 
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1
                    
                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1
                    
                        rs += 1
                    
                        if(pq > rs):
                            continue

                        termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))
                                                                      
                        termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)),  1/2.0)
                        termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)),  1/2.0)
                        termB += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), -1/2.0)
                        termB += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), -1/2.0)
 
                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)
               
                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)
                        
                        #Normalize
                        coeffA = 0
                        coeffB = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t

                        
                        if termA.many_body_order() > 0:
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)
                        
                        if termB.many_body_order() > 0:
                            termB = termB/np.sqrt(coeffB)
                            self.fermi_ops.append(termB)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 
# }}}

class GSD_notspinadapt(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        
        print(" Form non-spin adapted GSD operators")
        
        self.fermi_ops = []
        for p in range(0,2*self.n_orb):
 
            for q in range(p+1,2*self.n_orb):
        
                termA =  FermionOperator(((p,1),(q,0)))
                termA -= hermitian_conjugated(termA)
                termA = normal_ordered(termA)
                
                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t
            
                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)

        self.n_t1 = len(self.fermi_ops)
        print(type(self.fermi.ops))

        pq = -1 
        for p in range(0,2*self.n_orb):
 
            for q in range(p+1,2*self.n_orb):
        
                pq += 1
        
                rs = -1 
                for r in range(0,2*self.n_orb):
                    
                    for s in range(r+1,2*self.n_orb):
                    
                        rs += 1
                    
                        if(pq > rs):
                            continue

                        termA =  FermionOperator(((r,1),(p,0),(s,1),(q,0)), 1.0)
                        termA -= hermitian_conjugated(termA)
               
                        termA = normal_ordered(termA)
                        
                        #Normalize
                        coeffA = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        
                        if termA.many_body_order() > 0:
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)
                        
        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 
# }}}


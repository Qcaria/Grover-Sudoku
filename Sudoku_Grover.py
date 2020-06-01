import numpy as np
import scipy.linalg as la
from scipy.linalg import expm
import itertools
import operator
from projectq import MainEngine
from projectq.ops import X, Y, Z, H,  S, T, CX, CZ, Rx, Ry, Rz, MatrixGate, CRz, C, Measure, All, QubitOperator, TimeEvolution
from projectq.meta import Loop, Compute, Uncompute, Control
from projectq.cengines import (MainEngine,
                               AutoReplacer,
                               LocalOptimizer,
                               TagRemover,
                               InstructionFilter,
                               DecompositionRuleSet)
import projectq.setups.decompositions
import projectq.setups.decompositions as rules
from projectq.backends import Simulator
from hiq.projectq.backends import SimulatorMPI
from hiq.projectq.cengines import GreedyScheduler, HiQMainEngine

#Two extra libraries used by the functions implemented
from sympy.utilities.iterables import multiset_permutations
import math

# Define functions that will be used
#----------------------------------------------------------------------------------

def oracle(eng, qubits, bit_string, a):
    '''
    Implements an oracle that acts on ancilla 'a' if the state is the one given by bit_string.
    Args:
        eng (MainEngine): Main compiler engine the algorithm is being run on.
        qubits (Qureg): n-qubit quantum register Grover search is run on.
        a (Qubit): Ancilla qubit to flip in order to mark that the state satisfies the condition.
    '''
    with Compute(eng):
        for i in range(len(bit_string)):
            if not int(bit_string[i]): X | qubits[i]
    with Control(eng, qubits):
        X | a
    Uncompute(eng)

def get_perms(l_num):
    '''
    Computes all the possible states that some qubits can be in,
    given the digits in their file, column or box.
    '''
    _aux = [0, 1, 2, 3]
    aux = [n for n in _aux if n not in l_num]
    return [k for k in multiset_permutations(aux)]
  
def l_to_bin(l_perms):
    '''
    Transforms a list of digits into a suitable binary string to be implemented as an oracle
    '''
    l_bin = []
    for el in l_perms:
        l_bin.append(''.join([format(n, '#04b')[2:] for n in el]))
    return l_bin
  
def bin_to_n(bit_string):
    '''
    Transforms a binary string back to a list of decimal numbers to be shown as our solution
    '''
    n_list = []
    for i in range(int(len(bit_string)/2)):
        n_list.append(int(bit_string[2*i:2*i+2], 2))
    return n_list

def prep_sudoku(sudoku):
    '''
    Given a sudoku string, returns a list of groups (rows, columns or boxes) that are suitable to apply the sudoku rules.
    The suitability conditions are: No blanck spaces ('b' character), a length of 4 squares and at least one slot in the group.
    '''
    #Prepare the sudoku in list form
    new = sudoku.split('\n')
    _groups = []
    min_len = 4
    
    for i in range(len(new)):
        new[i] = new[i].split(',')
        #Clean possible unwanted elements '' and ' '
        new[i] = list(filter(lambda x: x not in ['', ' '], new[i]))
        
        #Adds each row to the group aspirants list
        _groups.append(new[i])
        
        #Gets the length of the shorter row so we dont exceed the index when computing columns.
        if len(new[i])<min_len: min_len = len(new[i])
        
    for i in range(min_len):
        col = [k[i] for k in new]
        #Adds columns to the group aspirants list
        _groups.append(col)

    #Computes the existing full boxes in the sudoku:
    for i in range(2):
        for j in range(2):
            if len(new)>=(j+1)*2:
                if len(new[i*2])>=(i+1)*2 and len(new[i*2+1])>=(i+1)*2:
                    box = [new[j*2][i*2], new[j*2][i*2+1], new[j*2+1][i*2], new[j*2+1][i*2+1]]
                    #Adds boxes to the group aspirants list
                    _groups.append(box)
       
    #Filters the groups given the criteria explained at the beginning of the function
    groups = list(filter(lambda el: len(el)==4 and 'b' not in el and 'x' in [c[0] for c in el], _groups))
    return groups

def apply_oracles(eng, groups, qudic, ancillas):
    '''
    Implements all the small oracles given the permutation groups list
    Args:
        eng (MainEngine): Main compiler engine the algorithm is being run on.
        groups(list): List of groups. Each group is a set of 4 elements where we can apply sudoku rules because they are members of the same row, columns or box.
        qudic (dictionary): Dictionary that assigns each slot (for example x1) to its corresponding two qubits.
        ancillas (Qureg): Quantum register containing the ancillas that are used to mark states that satisfy the rules of one group. Each group has its own ancilla.
    '''
    ind = 0
    for group in groups:
        digits = []
        qubits = []
        #Separate the digits and the corresponding qubits of the group.
        for el in group:
            if el[0] == 'x':
                qubits.append(qudic[el])
            else: digits.append(int(el))
                
        #Flatten the qubits list
        qubits = [q for sub in qubits for q in sub]
        
        #Gets all the possible states of the qubits given the group and generates the corresponding oracles, that act on the ancilla corresponding to that group.
        for bin_perm in l_to_bin(get_perms(digits)):
            oracle(eng, qubits, bin_perm, ancillas[ind])
        ind += 1


def SolveSudoku(eng,sudoku):
    '''
    Main function. Applies the grover algorithm one time to the qubits and returns the measured result. This result will be the solution of the sudoku with high probability.
    Args:
        eng (MainEngine): Main compiler engine the algorithm is being run on.
        sudoku (string): Sudoku codified in a string.
    '''
    #Prepare a list of suitable groups and a list of slot names, identified by an 'x'.
    groups = prep_sudoku(sudoku)
    l_slots = sorted(['x'+sudoku[i+1] for i in range(len(sudoku)) if sudoku[i] == 'x'], key = lambda n: int(n[1]))

    #Prepare a dictionary that assigns two qubits to each slot.
    qudic = {}
    for slot in l_slots:
        qudic[slot] = eng.allocate_qureg(2)
        q_list = [q for key in qudic.keys() for q in qudic[key]]

    #Compute the number of Grover iterations
    num_it = int(math.pi / 4. * math.sqrt(1 << len(q_list)))
    
    #Create one ancilla per group
    ancillas = eng.allocate_qureg(len(groups))

    #Prepare the oracle qubit, used to mark with a negative phase the sudoku solution
    phase_q = eng.allocate_qubit()
    X | phase_q
    H | phase_q
    
    #Start the superposition state
    All(H) | q_list
    
    with Loop(eng, num_it):
        #Generate the oracles and connect them to their group ancilla.
        with Compute(eng):
            apply_oracles(eng, groups, qudic, ancillas)
        
        #Most important step. Applies a toffoli gate between all the ancillas and the phase qubit. The state that fulfills the sudoku rules for all the groups, and thus solves the sudoku, will get a minus sign.
        with Control(eng, ancillas):
            X | phase_q 
        #Uncompute to clean the ancillas for future iterations
        Uncompute(eng)
  
        #Apply Grover diffusion operator
        All(H) | q_list
        All(X) | q_list
        with Control(eng, q_list):
            X | phase_q
        All(X) | q_list
        All(H) | q_list

    #Measure everything
    All(Measure) | q_list
    All(Measure) | ancillas
    Measure | phase_q
    
    # Call the main engine to execute
    eng.flush()

    # Obtain the output. The measured result will be the solution with high probability.
    results = ''
    for q in q_list:
        results = results+str(int(q))
        
    #Translate the bit string to a list of numbers
    xlist = bin_to_n(results)
    return xlist

# Run the program
#-----------------------------------------------------------------------------------------
backend = Simulator(gate_fusion=True)
cache_depth = 10
rule_set = DecompositionRuleSet(modules=[rules])
engines = [TagRemover(), LocalOptimizer(cache_depth), AutoReplacer(rule_set)]
eng = HiQMainEngine(backend, engines)

sudoku = 'x1,0,b,b\n,2,1,b,b\n,x2,3,b,b\n1,x3,3,x4'
#sudoku4x4 =  '2,1,0,3\n,0,x1,x2,1\n,1,x3,x4,x5\n3,0,1,2'
xlist = SolveSudoku(eng,sudoku)

print(xlist)

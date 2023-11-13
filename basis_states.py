import qiskit
from symmer import PauliwordOp, QuantumState
from symmer.utils import exact_gs_energy
import itertools
from hamiltonian import H, string_replace
import numpy as np

def B(number_of_lattice_sites):
	#Baryon number operator 
	B =  PauliwordOp.empty(2*number_of_lattice_sites)
	identity = 'I' * 2 * number_of_lattice_sites

	for i in range(0, 2*number_of_lattice_sites):
		B +=  PauliwordOp.from_dictionary({string_replace(identity, i, 'Z'): 1.0})

	B = B.multiply_by_constant(1/4)
	return B

def Qx_tot(number_of_lattice_sites):
	q =  PauliwordOp.empty(2 * number_of_lattice_sites)
	identity = 'I' * 2 * number_of_lattice_sites
	plus_coeffs = [1/2, 1j/2]
	minus_coeffs = [1/2, -1j/2]

	for n in range(1, number_of_lattice_sites + 1):
		plus = [string_replace(identity, 2*n - 2, 'X'), string_replace(identity, 2*n - 2, 'Y')]
		minus = [string_replace(identity, 2*n - 1, 'X'), string_replace(identity, 2*n - 1, 'Y')]

		P1 =  PauliwordOp.from_list(plus, plus_coeffs)
		P2 =  PauliwordOp.from_list(minus, minus_coeffs)

		op = P1 * P2
		q += (op + op.dagger)
       
	return q.multiply_by_constant(1/2)

def Qy_tot(number_of_lattice_sites):
	q =  PauliwordOp.empty(2 * number_of_lattice_sites)
	identity = 'I' * 2 * number_of_lattice_sites
	plus_coeffs = [1/2, 1j/2]
	minus_coeffs = [1/2, -1j/2]

	for n in range(1, number_of_lattice_sites + 1):
		minus = [string_replace(identity, 2*n - 2, 'X'), string_replace(identity, 2*n - 2, 'Y')]
		plus = [string_replace(identity, 2*n - 1, 'X'), string_replace(identity, 2*n - 1, 'Y')]

		P1 =  PauliwordOp.from_list(minus, minus_coeffs)
		P2 =  PauliwordOp.from_list(plus, plus_coeffs)

		op = P1 * P2
		q += (op - op.dagger)
       
	return q.multiply_by_constant(1j/2)

def Qz_tot(number_of_lattice_sites):
	q =  PauliwordOp.empty(2 * number_of_lattice_sites)
	identity = "I" * 2 * number_of_lattice_sites

	for n in range(1, number_of_lattice_sites + 1):
		Pz1 =  PauliwordOp.from_dictionary({string_replace(identity, 2*n - 2, 'Z'): 1.0})
		Pz2 =  PauliwordOp.from_dictionary({string_replace(identity, 2*n - 1, 'Z'): 1.0})

		q += (Pz1 - Pz2)
	return q.multiply_by_constant(1/4)

def is_physical_state(quantum_state, number_of_lattice_sites):
	qs_prime = Qz_tot(number_of_lattice_sites) * quantum_state
	return list((qs_prime).to_dictionary.values())[0] == 0

def is_color_singlet(quantum_state, number_of_lattice_sites):
	#Here, we assume that the quantum state passed in is a physical state 
	#correspoinding to a particular sector (B = 0, 1)
	qs_prime_x = Qx_tot(number_of_lattice_sites) * quantum_state
	qs_prime_y = Qy_tot(number_of_lattice_sites) * quantum_state
	return list((qs_prime_x).to_dictionary.values())[0] == 0 and \
		list((qs_prime_y).to_dictionary.values())[0] == 0


def get_basis(number_of_lattice_sites):
	comp_basis = [list(i) for i in itertools.product([0, 1], repeat = 2 * number_of_lattice_sites)]

	meson_basis = []
	baryon_basis = []


	for state in comp_basis:
		qs =  QuantumState([state])
		baryon_number = qs.dagger * B(number_of_lattice_sites) * qs

		if is_physical_state(qs, number_of_lattice_sites):
			if baryon_number == 1.0:
				baryon_basis.append(qs)
			elif baryon_number == 0.0:
				meson_basis.append(qs)
	return meson_basis, baryon_basis


def one_hot_to_binary_encoding(states):
    new_nqubits = int(np.ceil(np.log2(len(states))))
    return {format(i, '0' + str(new_nqubits) + 'b'):states[i] for i in range(len(states))}

def H_sector(hamiltonian, states):
    size = len(states)
    H_sector_matrix = np.zeros((size, size), dtype = complex)
    
    for i in range(len(states)):
        for j in range(len(states)):
            H_sector_matrix[i][j] = states[i].dagger * hamiltonian * states[j]
    return PauliwordOp.from_matrix(H_sector_matrix, disable_loading_bar=True)

def revert_to_full_basis(state, encoding_dictionary, number_full_qubits):
    psi_OH = QuantumState([0] * number_full_qubits)*0

    for psi_state in state.to_dictionary:
        psi_OH += encoding_dictionary[psi_state] * state.to_dictionary[psi_state]
    
    return psi_OH

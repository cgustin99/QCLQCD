import qiskit
import symmer as sym
from symmer.utils import exact_gs_energy
import itertools
from hamiltonian import H, string_replace
import numpy as np

def B(number_of_lattice_sites):
	#Baryon number operator 
	B = sym.PauliwordOp.empty(2*number_of_lattice_sites)
	identity = 'I' * 2 * number_of_lattice_sites

	for i in range(0, 2*number_of_lattice_sites):
		B += sym.PauliwordOp.from_dictionary({string_replace(identity, i, 'Z'): 1.0})

	B = B.multiply_by_constant(1/4)
	return B

def Qx_tot(number_of_lattice_sites):
	q = sym.PauliwordOp.empty(2 * number_of_lattice_sites)
	identity = 'I' * 2 * number_of_lattice_sites
	plus_coeffs = [1/2, 1j/2]
	minus_coeffs = [1/2, -1j/2]

	for n in range(1, number_of_lattice_sites + 1):
		plus = [string_replace(identity, 2*n - 2, 'X'), string_replace(identity, 2*n - 2, 'Y')]
		minus = [string_replace(identity, 2*n - 1, 'X'), string_replace(identity, 2*n - 1, 'Y')]

		P1 = sym.PauliwordOp.from_list(plus, plus_coeffs)
		P2 = sym.PauliwordOp.from_list(minus, minus_coeffs)

		op = P1 * P2
		q += (op + op.dagger)
       
	return q.multiply_by_constant(1/2)

def Qy_tot(number_of_lattice_sites):
	q = sym.PauliwordOp.empty(2 * number_of_lattice_sites)
	identity = 'I' * 2 * number_of_lattice_sites
	plus_coeffs = [1/2, 1j/2]
	minus_coeffs = [1/2, -1j/2]

	for n in range(1, number_of_lattice_sites + 1):
		minus = [string_replace(identity, 2*n - 2, 'X'), string_replace(identity, 2*n - 2, 'Y')]
		plus = [string_replace(identity, 2*n - 1, 'X'), string_replace(identity, 2*n - 1, 'Y')]

		P1 = sym.PauliwordOp.from_list(minus, minus_coeffs)
		P2 = sym.PauliwordOp.from_list(plus, plus_coeffs)

		op = P1 * P2
		q += (op - op.dagger)
       
	return q.multiply_by_constant(1j/2)

def Qz_tot(number_of_lattice_sites):
	q = sym.PauliwordOp.empty(2 * number_of_lattice_sites)
	identity = "I" * 2 * number_of_lattice_sites

	for n in range(1, number_of_lattice_sites + 1):
		Pz1 = sym.PauliwordOp.from_dictionary({string_replace(identity, 2*n - 2, 'Z'): 1.0})
		Pz2 = sym.PauliwordOp.from_dictionary({string_replace(identity, 2*n - 1, 'Z'): 1.0})

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
		qs = sym.QuantumState([state])
		baryon_number = qs.dagger * B(number_of_lattice_sites) * qs

		if is_physical_state(qs, number_of_lattice_sites):
			if baryon_number == 1.0:
				baryon_basis.append(qs)
			elif baryon_number == 0.0:
				meson_basis.append(qs)
	return meson_basis, baryon_basis

def color_singlets(number_of_lattice_sites, full_basis):

	#TODO: Baryon states for N > 4 and meson states for N > 2
	#		not yet color neutral

	basis = full_basis[:]
	basis_strings = [list(basis[i].to_dictionary.keys())[0] for i in range(len(basis))]

	color_singlets = []

	num_states_used = 0
	states_used = []

	#Remove initial color singlets
	for state in basis:
		if is_color_singlet(state, number_of_lattice_sites):
			color_singlets.append(state)
			states_used.append(list(state.to_dictionary.keys())[0])
			basis.remove(state)
			num_states_used += 1

	for states in list(itertools.combinations(basis, r = 2)):
		if is_color_singlet(states[0] - states[1], number_of_lattice_sites):
			color_singlets.append((states[0] - states[1]).normalize)
			num_states_used += 2
			states_used.append(list(states[0].to_dictionary.keys())[0])
			states_used.append(list(states[1].to_dictionary.keys())[0])

	print(set(states_used) ^ set(basis_strings))
	return color_singlets

def H_sector(number_of_lattice_sites, sector):
	'''
	Returns the Hamiltonian in the sector of interest
	return_type: 'matrix', 'paulii' (what form you want the hamiltonian in)
	sector: (str) 'meson', 'baryon'
	'''
	if sector == 'meson':
		basis_states = get_basis(number_of_lattice_sites)[0]
	else: basis_states = get_basis(number_of_lattice_sites)[1]
	comp_basis = [list(i) for i in itertools.product([0, 1], repeat = 2 * number_of_lattice_sites)]

	ham = H(number_of_lattice_sites)
	ham_matrix = np.zeros((2**(2*N), 2**(2*N)), dtype = complex)

	#FULL MATRIX
	for i in range(2**(2*N)):
		for j in range(2**(2*N)):
			state_i = sym.QuantumState([comp_basis[i]])
			state_j = sym.QuantumState([comp_basis[j]])
			if state_i in basis_states and state_j in basis_states:
				matrix_el = state_i.dagger * ham * state_j
				ham_matrix[i][j] = matrix_el	
		#ADD 1E6 TO ALL NON-ZERO DIAGONAL ELEMENTS 
		if ham_matrix[i][i] == 0:
			ham_matrix[i][i] = 1e6	

	return sym.PauliwordOp.from_matrix(ham_matrix)

# meson_states, baryon_states = get_basis(N)

# meson_color_singlets = color_singlets(N, meson_states)
# print("Total # of meson states: ", len(meson_states))

# state1 = sym.QuantumState.from_dictionary({'01100110': 1.0})
# state2 = sym.QuantumState.from_dictionary({'10011001': 1.0})
# state3 = sym.QuantumState.from_dictionary({'10010110': 1.0})
# state4 = sym.QuantumState.from_dictionary({'01011010': 1.0})
# state5 = sym.QuantumState.from_dictionary({'10100101': 1.0})
# state6 = sym.QuantumState.from_dictionary({'01101001': 1.0})

-state1-state2+state3-state4-state5+state6

-0.250+0.000j |01010101> +
  0.000-0.250j |01010110> +
  0.000+0.250j |01011001> +
 -0.250+0.000j |01011010> +
  0.000+0.250j |01100101> +
 -0.250+0.000j |01100110> +
  0.250+0.000j |01101001> +
  0.000+0.250j |01101010> +
  0.000-0.250j |10010101> +
  0.250+0.000j |10010110> +
 -0.250+0.000j |10011001> +
 -0.000-0.250j |10011010> +
 -0.250+0.000j |10100101> +
 -0.000-0.250j |10100110> +
  0.000+0.250j |10101001> +
 -0.250+0.000j |10101010>,

# print(is_color_singlet(state1 + state2 + state3, N))


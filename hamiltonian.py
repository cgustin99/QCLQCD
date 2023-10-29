import qiskit
import numpy as np
import symmer as sym

def string_replace(string, index, replaced_value):
    return string[:index] + replaced_value + string[index + 1:]


def H_mass_term(number_of_lattice_sites):
    number_of_qubits = 2 * number_of_lattice_sites

    H_m = sym.PauliwordOp.empty(number_of_qubits)

    for n in range(1, number_of_lattice_sites + 1):
        P1 = ['I'] * number_of_qubits
        P2 = ['I'] * number_of_qubits

        P1[2*n - 2] = 'Z'
        P2[2*n - 1] = 'Z'
        coeff = (-1)**n / 2
        H_m += sym.PauliwordOp.from_dictionary({"".join(P1): coeff, "".join(P2): coeff})

    H_m += sym.PauliwordOp.from_dictionary({"I"* 2*number_of_lattice_sites: number_of_lattice_sites})
    return H_m


def H_kin_term(number_of_lattice_sites):
    number_of_qubits = 2 * number_of_lattice_sites
    H_kin = sym.PauliwordOp.empty(number_of_qubits)
  
    identity = 'I' * number_of_qubits
    plus_coeffs = [1/2, 1j/2]
    minus_coeffs = [1/2, -1j/2]

    for n in range(1, number_of_lattice_sites):
        plus1 = [string_replace(identity, 2*n - 2, 'X'), string_replace(identity, 2*n - 2, 'Y')]
        minus1 = [string_replace(identity, 2*n, 'X'), string_replace(identity, 2*n, 'Y')]

        plus2 = [string_replace(identity, 2*n - 1, 'X'), string_replace(identity, 2*n - 1, 'Y')]
        minus2 = [string_replace(identity, 2*n + 1, 'X'), string_replace(identity, 2*n + 1, 'Y')]
        

        Pplus1 = sym.PauliwordOp.from_list(plus1, plus_coeffs)
        Pminus1 = sym.PauliwordOp.from_list(minus1, minus_coeffs)
        Pz1 = sym.PauliwordOp.from_dictionary({string_replace(identity, 2*n - 1, 'Z'): 1.0})

        Pplus2 = sym.PauliwordOp.from_list(plus2, plus_coeffs)
        Pminus2 = sym.PauliwordOp.from_list(minus2, minus_coeffs)
        Pz2 = sym.PauliwordOp.from_dictionary({string_replace(identity, 2*n, 'Z'): 1.0})

        op = Pplus1 * Pminus1 * Pz1 + Pplus2 * Pminus2 * Pz2
        H_kin += (op + op.dagger)

    H_kin = H_kin.multiply_by_constant(-1/2)

    return H_kin

def H_electric_term(number_of_lattice_sites):
    number_of_qubits = 2 * number_of_lattice_sites

    H_el1 = sym.PauliwordOp.empty(number_of_qubits)
    H_el2 = sym.PauliwordOp.empty(number_of_qubits)
    H_el3 = sym.PauliwordOp.empty(number_of_qubits)

    identity = 'I' * number_of_qubits
    plus_coeffs = [1/2, 1j/2]
    minus_coeffs = [1/2, -1j/2]

    for n in range(1, number_of_lattice_sites):
        Pz1 = sym.PauliwordOp.from_dictionary({string_replace(identity, 2*n - 2, 'Z'): 1.0})
        Pz2 = sym.PauliwordOp.from_dictionary({string_replace(identity, 2*n - 1, 'Z'): 1.0})

        H_el1 += sym.PauliwordOp.from_dictionary({'I'*number_of_qubits: 1.0}) - (Pz1 * Pz2)
        H_el1 = H_el1.multiply_by_constant((number_of_lattice_sites - n))
    H_el1 = H_el1.multiply_by_constant(3/16)


    for n in range(1, number_of_lattice_sites - 1):
        for m in range(n + 1, number_of_lattice_sites):
            Pz1 = sym.PauliwordOp.from_dictionary({string_replace(identity, 2*n - 2, 'Z'): 1.0})
            Pz2 = sym.PauliwordOp.from_dictionary({string_replace(identity, 2*n - 1, 'Z'): 1.0})
            Pz3 = sym.PauliwordOp.from_dictionary({string_replace(identity, 2*m - 2, 'Z'): 1.0})
            Pz4 = sym.PauliwordOp.from_dictionary({string_replace(identity, 2*m - 1, 'Z'): 1.0})

            H_el2 += (Pz1 - Pz2) * (Pz3 - Pz4)
            H_el2 = H_el2.multiply_by_constant((number_of_lattice_sites - m))

            plus1 = [string_replace(identity, 2*n - 2, 'X'), string_replace(identity, 2*n - 2, 'Y')]
            minus1 = [string_replace(identity, 2*n - 1, 'X'), string_replace(identity, 2*n - 1, 'Y')]

            plus2 = [string_replace(identity, 2*m - 1, 'X'), string_replace(identity, 2*m - 1, 'Y')]
            minus2 = [string_replace(identity, 2*m - 2, 'X'), string_replace(identity, 2*m - 2, 'Y')]

            Pplus1 = sym.PauliwordOp.from_list(plus1, plus_coeffs)
            Pminus1 = sym.PauliwordOp.from_list(minus1, minus_coeffs)
            Pplus2 = sym.PauliwordOp.from_list(plus2, plus_coeffs)
            Pminus2 = sym.PauliwordOp.from_list(minus2, minus_coeffs)

            op = Pplus1 * Pminus1 * Pplus2 * Pminus2
            H_el3 += op + op.dagger

            H_el3 = H_el3.multiply_by_constant((number_of_lattice_sites - m))

            

    H_el2 = H_el2.multiply_by_constant(1/16)
    H_el3 = H_el3.multiply_by_constant(1/2)

    H_el = H_el1 + H_el2 + H_el3
    return H_el 


def H(number_of_lattice_sites, a = 1, m = 1, g = 1):
    number_of_qubits = 2 * number_of_lattice_sites

    x = 1 / (a**2 * g**2)
    Hamiltonian = H_mass_term(number_of_lattice_sites).multiply_by_constant(a*m)+\
        H_kin_term(number_of_lattice_sites).multiply_by_constant(1/x) +\
        H_electric_term(number_of_lattice_sites)

    return Hamiltonian
    

import numpy as np
from fractions import Fraction

def parse_input(eq):
    eq = eq.replace(" ", "")
    left, right = eq.split("=")
    constant = Fraction(right)

    coeffs = {'x': 0, 'y': 0, 'z': 0}
    left = left.replace("-", "+-")
    terms = left.split("+")

    for term in terms:
        if term == "":
            continue
        if 'x' in term:
            coeff = term.replace("x", '')
            coeffs['x'] = Fraction(coeff) if coeff not in ["", "+"] else 1
            if coeff == "-":
                coeffs['x'] = -1
        elif 'y' in term:
            coeff = term.replace("y", '')
            coeffs['y'] = Fraction(coeff) if coeff not in ["", "+"] else 1
            if coeff == "-":
                coeffs['y'] = -1
        elif 'z' in term:
            coeff = term.replace("z", '')
            coeffs['z'] = Fraction(coeff) if coeff not in ["", "+"] else 1
            if coeff == "-":
                coeffs['z'] = -1

    return [coeffs['x'], coeffs['y'], coeffs['z']], constant


def gauss_elim(A, B):
    n = len(B)
    augmented = np.concatenate((A, B), axis=1)

    steps = []

    # Save initial matrix
    steps.append("Initial Augmented Matrix:")
    steps.append([[str(val) for val in row] for row in augmented])

    # Forward elimination
    for i in range(n):
        for j in range(i+1, n):
            factor = augmented[j][i] / augmented[i][i]
            augmented[j] -= factor * augmented[i]

            steps.append(f"R{j+1} = R{j+1} - ({factor})R{i+1}")
            steps.append([[str(val) for val in row] for row in augmented])

    # Back substitution
    x = np.zeros(n, dtype=object)
    for i in range(n-1, -1, -1):
        sum_ax = sum(augmented[i][j] * x[j] for j in range(i+1, n))
        x[i] = (augmented[i][n] - sum_ax) / augmented[i][i]

    return x, steps
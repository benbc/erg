from sympy import Matrix, symbols, Eq, Integer, eye
from tabulate import tabulate

from lib import vector, dot, sum_vector, hadamard, hadamard_inv, solve, only, vector_to_tuple, round_all


def run(omega):
    """
    Solve the model for a given wage rate (omega).
    """
    # Free variables
    #
    # Naming conventions:
    #   - matrices are uppercase
    #   - vectors are lowercase
    #   - scalars are greek lowercase or indexed roman lowercase where they appear as vector components
    p1, p3, rho, w1, w2, k0, k1, k2, k3, kappa = symbols('p1 p3 rho w1 w2 k0 k1 k2 k3 kappa', negative=False, real=True)

    # Process definition. These are the numerical inputs to the model along with omega.
    A = Matrix([[0.3, 0.2, 0], [0, 0.4, 0.2], [1, 0, 0]])  # Inputs.
    I = eye(3)
    assert (I - A).det() >= 0, "Production matrix is not productive"  # Kurz and Salvadori eq 4.4
    l = vector(1, 0.25, 0.6)  # Labour required.

    p = vector(p1, 1, p3)  # Prices.
    w = vector(w1, w1, 0)  # Wage basket. Composition isn't fully determined: pick one where the amounts are equal.
    # TODO: what is the effect of varying wage basket composition?
    kappa = Integer(100)  # Population size.
    k = vector(k1, k2, k3)  # Sector labour assignments.
    k0 = kappa - sum_vector(k)  # Unassigned labour.
    u = hadamard(k, hadamard_inv(l))  # Production volume.
    # rho: Profit rate.

    # Solve the profit equation to determine non-numeraire price(s) and profit rate.
    profit_eq = Eq(p, (1 + rho) * (A * p + omega * l))
    # Kurz and Salvadori have this form instead (eq 4.6a). Which is right? Replicating Hahnel's profits requires
    # my version.
    # profit_eq = Eq(p, (1 + rho) * A * p + omega * l)
    rho_val, p1_val, p3_val = only(solve(profit_eq, rho, p1, p3))
    rho = rho.subs(rho, rho_val)
    p = p.subs(p1, p1_val).subs(p3, p3_val)

    # Solve the wage equation to determine the wage basket composition.
    wage_eq = Eq(dot(w, p), omega)
    w1_val = only(solve(wage_eq, w1))
    w = w.subs(w1, w1_val)

    # Solve the production equation to determine the labour assignments.
    production_eq = Eq(u, A.T * u + kappa * w)
    k1_val, k2_val, k3_val = only(solve(production_eq, k1, k2, k3))
    k = k.subs(k1, k1_val).subs(k2, k2_val).subs(k3, k3_val)
    k0 = k0.subs(k1, k1_val).subs(k2, k2_val).subs(k3, k3_val)

    return p, rho, w, k, k0


def main():
    table = []
    headers = ["wage rate", "prices", "profit", "wages", "assignments", "unassigned"]

    # These are almost Hahnel's examples. He's rounding his numbers, though, and his 0.691 actually gives a tiny
    # negative number which our model rejects.
    for o in [0.6909, 0.5, 0.4]:
        p, rho, w, k, k0 = run(o)
        table.append(round_all([o, vector_to_tuple(p), rho, vector_to_tuple(w), vector_to_tuple(k), k0]))

    print(tabulate(table, headers=headers))


if __name__ == "__main__":
    main()

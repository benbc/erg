from sympy import Matrix, symbols, Eq, Integer

from lib import vector, print_vector, dot, sum_vector, hadamard, hadamard_inv, solve_, only


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
    p1, rho, w1, w2, k1, k2, k0, kappa = symbols('p1 rho w1 w2 k1 k2 k0 kappa', negative=False)

    # Process definition. These are the numerical inputs to the model along with omega.
    A = Matrix([[0.3, 0.2], [0.2, 0.4]])  # Inputs.
    l = vector(1, 0.5)  # Labour required.

    p = vector(p1, 1)  # Prices.
    w = vector(w1, w1)  # Wage basket. Composition isn't fully determined: pick one where the amounts are equal.
    # TODO: what is the effect of varying wage basket composition?
    kappa = Integer(100)  # Population size.
    k = vector(k1, k2)  # Sector labour assignments.
    k0 = kappa - sum_vector(k)  # Unassigned labour.
    u = hadamard(k, hadamard_inv(l))  # Production volume.
    # rho: Profit rate.

    print(f"wage rate: {omega}")

    # Solve the profit equation to determine non-numeraire price(s) and profit rate.
    profit_eq = Eq(p, (1 + rho) * (A * p + omega * l))
    # Kurz and Salvadori have this form instead (eq 4.6a). Which is right? Replicating Hahnel's profits requires
    # my version.
    # profit_eq = Eq(p, (1 + rho) * A * p + omega * l)
    p1_val, rho_val = only(solve_(profit_eq, p1, rho))
    p = p.subs(p1, p1_val)
    print_vector("prices", p)
    print(f"profit: {rho_val}")

    # Solve the wage equation to determine the wage basket composition.
    wage_eq = Eq(dot(w, p), omega)
    w1_val = only(solve_(wage_eq, w1))
    w = w.subs(w1, w1_val)
    print_vector("wage basket", w)

    # Solve the production equation to determine the labour assignments.
    production_eq = Eq(u, A.T * u + kappa * w)
    k1_val, k2_val = only(solve_(production_eq, k1, k2))
    k = k.subs(k1, k1_val).subs(k2, k2_val)
    k0 = k0.subs(k1, k1_val).subs(k2, k2_val)
    print_vector("assignments", k)
    print(f"unassigned: {k0}")


if __name__ == "__main__":
    # These are almost Hahnel's examples. He's rounding his numbers, though, and his 0.691 actually gives a tiny
    # negative number which our model rejects.
    for o in [0.6909, 0.5, 0.4]:
        run(o)
        print()

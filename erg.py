from sympy import Matrix, symbols, Eq, Integer

from lib import vector, print_vector, dot, sum_vector, hadamard, hadamard_inv, _solve, only


def run(omega):
    p1, rho, w1, w2, k1, k2, k0, kappa = symbols('p1 rho w1 w2 k1 k2 k0 kappa', negative=False)
    rho, k1, k2, u1, u2 = symbols('rho k1 k2 u1 u2')

    p = vector(p1, 1)
    w = vector(w1, w1)  # Wage basket is unconstrained. For now, pick one where the amounts are equal.

    A = Matrix([[0.3, 0.2], [0.2, 0.4]])
    l = vector(1, 0.5)

    kappa = Integer(100)
    k = vector(k1, k2)
    k0 = kappa - sum_vector(k)

    u = hadamard(k, hadamard_inv(l))

    print(f"wage rate: {omega}")

    profit_eq = Eq(p, (1 + rho) * (A * p + omega * l))
    # Kurz and Salvadori have this form instead (eq 4.6a). Which is right? Replicating Hahnel's profits requires
    # my version.
    # profit_eq = Eq(p, (1 + rho) * A * p + omega * l)
    p1_val, rho_val = only(_solve(profit_eq, p1, rho))
    p = p.subs(p1, p1_val)
    print_vector("prices", p)
    print(f"profit: {float(rho_val)}")

    wage_eq = Eq(dot(w, p), omega)
    w1_val = only(_solve(wage_eq, w1))
    w = w.subs(w1, w1_val)
    print_vector("wage basket", w)
    # TODO: For now we've arbitrarily chosen a wage basket where the amounts are equal. Need to test what effect the
    #       composition has on the rest of the model below.

    production_eq = Eq(u, A.T * u + kappa * w)
    k1_val, k2_val = only(_solve(production_eq, k1, k2))
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

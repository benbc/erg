from sympy import Matrix, shape, symbols, Eq, solve, Integer


def vector(x1, x2):
    return Matrix([x1, x2])


def print_vector(prefix, v):
    assert shape(v)[1] == 1
    print(f"{prefix}: ({', '.join(str(x) for x in v.col(0))})")


def dot(v1, v2):
    hv1, wv1 = shape(v1)
    hv2, wv2 = shape(v2)
    assert wv1 == 1
    assert wv2 == 1
    assert hv1 == hv2
    mul = v1.T * v2
    assert shape(mul) == (1, 1)
    return mul.row(0)[0]


def sum_vector(v):
    assert shape(v)[1] == 1
    return sum(v.col(0))


def hadamard(m1, m2):
    assert shape(m1) == shape(m2)
    num_rows, num_columns = shape(m1)
    rows = []
    for i in range(num_rows):
        row = []
        for j in range(num_columns):
            row.append(m1.row(i)[j] * m2.row(i)[j])
        rows.append(row)
    return Matrix(rows)


def hadamard_inv(m):
    num_rows, num_columns = shape(m)
    rows = []
    for i in range(num_rows):
        row = []
        for j in range(num_columns):
            row.append(1 / m.row(i)[j])
        rows.append(row)
    return Matrix(rows)


def _solve(equation, *variables):
    def convert_dict(sol):
        return tuple(sol[var] for var in variables)

    solutions = solve(equation, *variables)

    if isinstance(solutions, tuple):
        return [solutions]
    if isinstance(solutions, dict):
        return [convert_dict(solutions)]

    assert isinstance(solutions, list)
    converted = []
    for solution in solutions:
        if isinstance(solution, tuple):
            converted.append(solution)
            continue
        if isinstance(solution, dict):
            converted.append(convert_dict(solution))
            continue
        converted.append(solution)
    return converted


def only(xs):
    assert len(xs) == 1, f"Not exactly one solution: {xs}"
    return xs[0]


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
    # Kurz and Salvadori have this form instead (eq 4.6a). Which is right?
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
    for o in [0.6909, 0.5, 0.4]:
        run(o)
        print()

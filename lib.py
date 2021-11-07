from sympy import Matrix, shape, solve


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


def solve_(equation, *variables):
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
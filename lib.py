from sympy import Matrix, shape, solve, Float


def vector(*xs):
    return Matrix(xs)


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


def vector_to_tuple(v):
    assert shape(v)[1] == 1
    return tuple(v.col(0))


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


def round_one(x):
    if isinstance(x, Float):
        # Ugh -- tabulate will call float() on these, which is broken. So we convert to a string here.
        # e.g. float(Float(0.0535068155343779).round(3)) => 0.05401611328125
        return str(x.round(3))
    if isinstance(x, float):
        return round(x, 3)
    return x


def round_all(xs):
    for x in xs:
        if isinstance(x, tuple):
            yield tuple(round_one(y) for y in x)
        else:
            yield round_one(x)

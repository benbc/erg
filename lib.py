from sympy import Matrix, shape, solve, Float


def same_shape(m1, m2):
    return shape(m1) == shape(m2)


def hadamard(m1, m2):
    assert same_shape(m1, m2)
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


def vector(*xs):
    return Matrix(xs)


def is_vector(v):
    return shape(v)[1] == 1


def print_vector(prefix, v):
    assert is_vector(v)
    print(f"{prefix}: ({', '.join(str(x) for x in v.col(0))})")


def sum_vector(v):
    assert is_vector(v)
    return sum(v.col(0))


def vector_to_tuple(v):
    assert is_vector(v)
    return tuple(v.col(0))


def dot(v1, v2):
    assert is_vector(v1)
    assert is_vector(v2)
    assert same_shape(v1, v2)
    return matrix_to_scalar(v1.T * v2)


def matrix_to_scalar(mul):
    assert is_scalar(mul)
    return mul.row(0)[0]


def is_scalar(mul):
    return shape(mul) == (1, 1)


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

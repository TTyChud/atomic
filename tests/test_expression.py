import numpy as np
import pytest

from atomic.numerics.expression import ExpressionError, compile_potential


def test_coulomb_expression_evaluates():
    f = compile_potential("-1/r")
    r = np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(f(r), [-1.0, -0.5, -0.25])

def test_oscillator_and_constants():
    f = compile_potential("0.5*r**2")
    np.testing.assert_allclose(f(np.array([2.0])), [2.0])
    g = compile_potential("pi")
    np.testing.assert_allclose(g(np.array([1.0, 9.0])), [np.pi, np.pi])

def test_whitelisted_functions_and_piecewise():
    f = compile_potential("-exp(-r)/r")
    np.testing.assert_allclose(f(np.array([1.0])), [-np.exp(-1.0)])
    well = compile_potential("where(r < 3, -2, 0)")
    np.testing.assert_allclose(well(np.array([1.0, 5.0])), [-2.0, 0.0])

@pytest.mark.parametrize(
    "expr",
    [
        '__import__("os")',
        "r.__class__",
        "foo(r)",
        "a + 1",
        "r[0]",
        "lambda r: r",
        "[x for x in r]",
        "r and 1",
    ],
)
def test_rejects_non_whitelisted(expr):
    with pytest.raises(ExpressionError):
        compile_potential(expr)

def test_dos_guards():
    with pytest.raises(ExpressionError):
        compile_potential("r**99")
    with pytest.raises(ExpressionError):
        compile_potential("r" * 201)
    with pytest.raises(ExpressionError):
        compile_potential("+".join(["r"] * 60))

def test_error_message_names_construct():
    with pytest.raises(ExpressionError, match="Attribute"):
        compile_potential("r.real")

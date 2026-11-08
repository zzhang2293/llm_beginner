"""Checks for the TODOs in coffee_prediction.ipynb.

Each check calls your function with known inputs and compares the result with
the correct answer. It prints one of these results:

  PASS      Your function gives the correct answer.
  FAIL      Your function gives a wrong answer or an error. Read the message.
  NOT DONE  A function still raises NotImplementedError.

You do not need to read or change this file.
"""

import os
import textwrap
import traceback

import numpy as np

# The trained weights. They are the same as the weights in the notebook.
_W1 = np.array([[-8.93, 0.29, 12.9], [-0.1, -7.32, 10.81]])
_B1 = np.array([-9.82, -9.28, 0.96])
_W2 = np.array([[-31.18], [-27.59], [-32.56]])
_B2 = np.array([15.41])

# The mean and the standard deviation of the training data.
_MEAN = np.array([218.6678330766, 13.4306848851])
_STD = np.array([39.9616562128, 1.1284171662])

# Three test coffees: [temperature in degrees Celsius, duration in minutes].
_NAMES = ["200 C, 13.9 min", "200 C, 17.0 min", "165 C, 13.0 min"]
_X_RAW = np.array([[200.0, 13.9], [200.0, 17.0], [165.0, 13.0]])
# The same three coffees after normalization.
_X_NORM = np.array([
    [-0.4671436283, 0.4159056854],
    [-0.4671436283, 3.1631166396],
    [-1.3429832035, -0.3816716884],
])
# The correct network output a2 for each test coffee.
_A2 = np.array([[9.7193037888e-01], [3.2897870199e-08], [2.9939155352e-06]])

# sigmoid([1, 2, 3.5]). Used to find a common mistake in dense().
_SIGMOID_OF_Z = np.array([0.7310585786, 0.8807970780, 0.9706877692])


class _Fail(Exception):
    """A check found a wrong answer."""


def check_normalize(normalize):
    """Checks TODO 1: normalize()."""

    def test():
        X = np.array([[250.0, 14.0], [300.0, 11.0]])
        X_before = X.copy()
        what = ("normalize(X, mean, std) with X = [[250, 14], [300, 11]], "
                "mean = [200, 13], std = [50, 1]")
        got = _call(what, normalize, X, np.array([200.0, 13.0]), np.array([50.0, 1.0]))
        _compare(what, got, [[1.0, 1.0], [2.0, -2.0]],
                 hint="Use the mean and std arguments. Do not compute them from X.")
        if not np.array_equal(X, X_before):
            raise _Fail("normalize() changed its input X. Make a new array. "
                        "Do not use -= or /= on X.")

        what = "normalize() for the test coffees [[200, 13.9], [200, 17], [165, 13]]"
        got = _call(what, normalize, _X_RAW, _MEAN, _STD)
        _compare(what, got, _X_NORM)

    _run("TODO 1: normalize()", test)


def check_sigmoid(sigmoid):
    """Checks TODO 2: sigmoid()."""

    def test():
        what = "sigmoid(0)"
        _compare(what, _call(what, sigmoid, 0.0), 0.5)

        what = "sigmoid(z) with z = np.array([-2, -1, 0, 1, 2])"
        got = _call(what, sigmoid, np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
                    hint="Use np.exp, not math.exp. np.exp works on a whole array.")
        _compare(what, got, [0.1192029220, 0.2689414214, 0.5, 0.7310585786, 0.8807970780])

        what = "sigmoid(z) with z = np.array([-30, 30])"
        got = _call(what, sigmoid, np.array([-30.0, 30.0]))
        _compare(what, got, [9.3576229688e-14, 1.0])

    _run("TODO 2: sigmoid()", test)


def check_dense(dense):
    """Checks TODO 3: dense()."""

    def no_activation(z):
        return z

    def test():
        a_in = np.array([1.0, 2.0])
        W = np.array([[1.0, -1.0, 0.5],
                      [0.0, 1.0, 2.0]])
        b = np.array([0.0, 1.0, -1.0])
        z = np.array([1.0, 2.0, 3.5])  # the correct z for each neuron

        what = ("dense(a_in, W, b, g) with a_in = [1, 2], W = [[1, -1, 0.5], [0, 1, 2]], "
                "b = [0, 1, -1], and g(z) = z (no activation)")
        got = _call(what, dense, a_in, W, b, no_activation,
                    hint="Neuron j uses column j of W, which is W[:, j].")
        if _same(got, z - b):
            raise _Fail(f"{what}\nThe bias is missing. Add b_j to z_j.")
        if _same(got, _SIGMOID_OF_Z):
            raise _Fail(f"{what}\ndense() used sigmoid. Use g, the activation "
                        "function that the caller gives.")
        if _same(got, [z[0], 0.0, 0.0]):
            raise _Fail(f"{what}\nOnly neuron 0 has a value. Is a return statement "
                        "inside the for loop? Return a_out after the loop.")
        _compare(what, got, z)

        what = ("dense() with 1 neuron: a_in = [1, 0, -1], W = [[2], [5], [1]], "
                "b = [0.5], and g(z) = z")
        got = _call(what, dense, np.array([1.0, 0.0, -1.0]),
                    np.array([[2.0], [5.0], [1.0]]), np.array([0.5]), no_activation)
        _compare(what, got, [1.5])

        what = "dense(a_in, W, b, g) with the same a_in, W, b, and g = np.tanh"
        got = _call(what, dense, a_in, W, b, np.tanh)
        if _same(got, z):
            raise _Fail(f"{what}\ndense() did not apply g. Use a_out[j] = g(z_j).")
        _compare(what, got, np.tanh(z))

    _run("TODO 3: dense()", test)


def check_forward(forward):
    """Checks TODO 4: forward()."""

    def test():
        for name, x, a2 in zip(_NAMES, _X_NORM, _A2):
            what = f"forward(x, W1, b1, W2, b2) for the coffee ({name})"
            got = _call(what, forward, x, _W1, _B1, _W2, _B2)
            if got is not None and np.shape(got) == (3,):
                raise _Fail(f"{what}\nforward() returned 3 values. Return a2, "
                            "the output of layer 2, with shape (1,).")
            if got is not None and np.shape(got) == ():
                raise _Fail(f"{what}\nforward() returned one number with shape (). "
                            "Return the array a2 with shape (1,), not a2[0].")
            _compare(what, got, a2)

    _run("TODO 4: forward()", test)


def check_predict(predict):
    """Checks TODO 5: predict()."""

    def test():
        what = "predict(X, W1, b1, W2, b2) for the 3 test coffees"
        got = _call(what, predict, _X_NORM, _W1, _B1, _W2, _B2)
        _compare(what, got, [[1], [0], [0]])

        what = "predict(X, W1, b1, W2, b2, threshold=0.99) for the 3 test coffees"
        got = _call(what, predict, _X_NORM, _W1, _B1, _W2, _B2, threshold=0.99)
        _compare(what, got, [[0], [0], [0]],
                 hint="Use the threshold argument, not a fixed 0.5.")

    _run("TODO 5: predict()", test)


def check_forward_vectorized(forward_vectorized):
    """Checks the bonus TODO: forward_vectorized()."""

    def test():
        what = "forward_vectorized(X, W1, b1, W2, b2) for the 3 test coffees"
        got = _call(what, forward_vectorized, _X_NORM, _W1, _B1, _W2, _B2,
                    hint="X @ W1 has shape (m, 3). NumPy adds b1 to each row.")
        _compare(what, got, _A2)

    _run("BONUS: forward_vectorized()", test)


def _run(title, test):
    """Runs one check and prints the result."""
    try:
        test()
    except NotImplementedError as error:
        print(f"NOT DONE  {title}\n{textwrap.indent(str(error), '  ')}")
    except _Fail as error:
        print(f"FAIL  {title}\n{textwrap.indent(str(error), '  ')}")
    else:
        print(f"PASS  {title}")


def _call(what, function, *args, hint=None, **kwargs):
    """Calls your function. If it raises an error, the check fails with a clear message."""
    try:
        return function(*args, **kwargs)
    except NotImplementedError:
        raise
    except Exception as error:  # Any error in your code is a FAIL.
        message = f"{what}\nYour code raised {type(error).__name__}: {error}"
        location = _location(error)
        if location:
            message += f"\n{location}"
        if hint:
            message += f"\nHint: {hint}"
        raise _Fail(message) from None


def _location(error):
    """Returns the lines of your code where the error happened."""
    this_file = os.path.abspath(__file__)
    frames = [
        frame for frame in traceback.extract_tb(error.__traceback__)
        if os.path.abspath(frame.filename) != this_file
        and "site-packages" not in frame.filename
    ]
    if not frames:
        return ""
    lines = [f"    in {frame.name}(): {(frame.line or '').strip()}" for frame in frames]
    return "The error happened here:\n" + "\n".join(lines)


def _compare(what, got, expected, hint=None):
    """Fails the check if got does not have the shape and the values of expected."""
    if got is None:
        raise _Fail(f"{what}\nYour function returned None. "
                    "Did you forget the return statement?")
    try:
        got = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        raise _Fail(f"{what}\nYour function returned a {type(got).__name__}, "
                    "not a number or a NumPy array.") from None
    expected = np.asarray(expected, dtype=float)

    if got.shape != expected.shape:
        message = (f"{what}\nThe shape is wrong.\n"
                   f"  expected shape: {expected.shape}\n"
                   f"  your shape:     {got.shape}")
    elif not np.allclose(got, expected, rtol=1e-6, atol=1e-9):
        message = (f"{what}\nThe values are wrong.\n"
                   f"  expected: {_format(expected)}\n"
                   f"  yours:    {_format(got)}")
    else:
        return
    if hint:
        message += f"\nHint: {hint}"
    raise _Fail(message)


def _same(got, expected):
    """Returns True if got has the shape and the values of expected."""
    try:
        got = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return False
    expected = np.asarray(expected, dtype=float)
    return got.shape == expected.shape and np.allclose(got, expected, rtol=1e-6, atol=1e-9)


def _format(array):
    """Formats an array on one line."""
    return np.array2string(array, precision=6, separator=", ").replace("\n", "")

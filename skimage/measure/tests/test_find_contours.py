import numpy as np
from skimage.measure import find_contours

from skimage._shared import testing
from skimage._shared.testing import assert_array_equal

# L-shaped example scalar field
L = np.ones((8, 8), dtype=np.float32)
L[1:-1, 1] = 0
L[1, 1:-1] = 0

# Resulting contour
target = [
    [6. ,  1.5],
    [6.5,  1. ],
    [6. ,  0.5],
    [5. ,  0.5],
    [4. ,  0.5],
    [3. ,  0.5],
    [2. ,  0.5],
    [1. ,  0.5],
    [0.5,  1. ],
    [0.5,  2. ],
    [0.5,  3. ],
    [0.5,  4. ],
    [0.5,  5. ],
    [0.5,  6. ],
    [1. ,  6.5],
    [1.5,  6. ],
    [1.5,  5. ],
    [1.5,  4. ],
    [1.5,  3. ],
    [1.5,  2. ],
    [2. ,  1.5],
    [3. ,  1.5],
    [4. ,  1.5],
    [5. ,  1.5],
    [6. ,  1.5],
]

mask = np.ones((8, 8), dtype=bool)
# Some missing data that should result in a hole in the contour:
mask[7, 0:2] = False

# Some missing data that shouldn't change anything:
mask[0, 7] = False
mask[2, 7] = False

x, y = np.mgrid[-1:1:5j, -1:1:5j]
r = np.sqrt(x**2 + y**2)

def assert_equal_loops(a, b):
    # Check whether two closed paths are equal.
    # They might have different starting points.
    assert np.shape(a) == np.shape(b), \
        "%r and %r have different shapes" % (a,b)
    assert np.all(a[0] == a[-1]), "%r is not a loop" % a
    assert np.all(b[0] == b[-1]), "%r is not a loop" % b
    aa = a[:-1]
    bb = b[:-1]
    for i in range(np.shape(aa)[0]):
        if np.allclose(aa, np.roll(bb, i, axis=0)):
            return
    raise AssertionError("%r and %r are not equal as loops" % (a,b))

def rot_contour(c, size=8):
    # rotate a list of indices by 90 degrees ccw, to match np.rot90 being
    # applied to the array they are indexing.
    # We are relying on the fact that L is square!
    return np.dot(c, [[0,1], [-1,0]]) + [size - 1, 0]

def test_binary():
    a = np.copy(L)
    ref = np.copy(target)
    for _ in range(4):
        contours = find_contours(a, 0.5, positive_orientation='high')
        assert len(contours) == 1
        assert_equal_loops(contours[0], ref)
        a = np.rot90(a)
        ref = rot_contour(ref)


def test_nodata():
    # Test missing data via NaNs in input array
    b = np.copy(L)
    b[~mask] = np.nan
    ref = np.copy(target)[2:]
    for _ in range(4):
        contours = find_contours(b, 0.5, positive_orientation='high')
        assert len(contours) == 1
        assert_array_equal(contours[0], ref)
        b = np.rot90(b)
        ref = rot_contour(ref)

def test_mask():
    # Test missing data via explicit masking
    a = np.copy(L)
    m = np.copy(mask)
    ref = np.copy(target)[2:]
    for _ in range(4):
        contours = find_contours(a, 0.5, positive_orientation='high', mask=m)
        assert len(contours) == 1
        assert_array_equal(contours[0], ref)
        a = np.rot90(a)
        m = np.rot90(m)
        ref = rot_contour(ref)


def test_float():
    contours = find_contours(r, 0.5)
    assert len(contours) == 1
    assert_array_equal(contours[0],
                    [[ 2.,  3.],
                     [ 1.,  2.],
                     [ 2.,  1.],
                     [ 3.,  2.],
                     [ 2.,  3.]])


def test_memory_order():
    contours = find_contours(np.ascontiguousarray(r), 0.5)
    assert len(contours) == 1

    contours = find_contours(np.asfortranarray(r), 0.5)
    assert len(contours) == 1


def test_invalid_input():
    with testing.raises(ValueError):
        find_contours(r, 0.5, 'foo', 'bar')
    with testing.raises(ValueError):
        find_contours(r[..., None], 0.5)

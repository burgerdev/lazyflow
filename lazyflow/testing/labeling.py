# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# Copyright 2011-2014, the ilastik developers

import numpy as np


## check if two label arrays are equivalent
# Even if two labeling operations generate the same connected components,
# the assignment of label values to objects is not specified and is
# expected to vary across implementations. This function checks whether
# two labelings differ in meaning, and raises an AssertionError if that
# is the case. Label 0 is ignored, but after all connected components
# have been checked the backgrounds have to be the same, too.
#
# @param labeling the labeling that shall be tested
# @param reference the reference array (ground truth)
def assertEquivalentLabeling(labeling, reference):
    x = labeling
    y = reference
    assert np.all(x.shape == y.shape),\
        "Shapes do not agree ({} vs {})".format(x.shape, y.shape)

    # identify labels used in x
    ref_labels = set(y.flat)
    x_labels = set(x.flat)
    assert len(ref_labels) == len(x_labels),\
        "Number of connected components is different (ref: {}, img: {})".format(len(ref_labels), len(x_labels))
    for label in ref_labels:
        if label == 0:
            # background agrees after all labels are checked
            continue

        # extract the region that was labeled in the reference image
        cc_idx = np.where(y == label)
        cc_in_x = x[cc_idx]

        # get a representative index for this CC
        an_index = [a[0] for a in cc_idx]
        x_label = cc_in_x[0]
        
        # check if the labeled block has exactly one label
        assert np.all(cc_in_x == x_label),\
            "component around {} has multiple labels, should be a single label".format(an_index)

        # check that nothing else is labeled with this label
        m = cc_in_x.size
        n = len(np.where(x == x_label)[0])
        assert m == n,\
            "Label {} is used for multiple components (expected {} pix, got {} pix).".format(x_label, m, n)


if __name__ == "__main__":
    x = np.asarray([[1, 1, 0, 0, 2],
                    [1, 0, 0, 2, 2],
                    [3, 0, 0, 0, 2],
                    [0, 4, 0, 5, 5]], dtype=np.int)
    assertEquivalentLabeling(x, x)
    assertEquivalentLabeling(3*x, x)
    
    y = x.copy()
    y[y == 2] = -1
    y[y == 3] = 2
    y[y == -1] = 3
    assertEquivalentLabeling(y, x)

    y[0, 2] = 1
    try:
        assertEquivalentLabeling(y, x)
    except AssertionError:
        pass
    else:
        assert False, "Wrong labeling 1 got through"

    z = np.asarray([[1, 1, 0, 0, 2],
                    [1, 0, 0, 6, 6],
                    [3, 0, 0, 0, 5],
                    [0, 4, 0, 5, 5]], dtype=np.int)
    try:
        assertEquivalentLabeling(z, x)
    except AssertionError:
        pass
    else:
        assert False, "Wrong labeling 2 got through"

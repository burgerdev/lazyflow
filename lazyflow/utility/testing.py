###############################################################################
#   lazyflow: data flow based lazy parallel computation framework
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the Lesser GNU General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# See the files LICENSE.lgpl2 and LICENSE.lgpl3 for full text of the
# GNU Lesser General Public License version 2.1 and 3 respectively.
# This information is also available on the ilastik web site at:
#          http://ilastik.org/license/
###############################################################################

import numpy as np


# Some tools to aid automated testing

# check that two label images have the same meaning
# (label images can differ in the actual labels, but cannot differ in their
# spatial structure)
#
# Examples:
#   Input:
#       1 1 0    250 250 0
#       1 1 0    250 250 0
#       0 0 0    0   0   0
#   Output: None
#
#   Input:
#       1 1 0    250 250 0
#       1 1 0    1   1   0
#       0 0 0    0   0   0
#   Output: AssertionError
#
def assertEquivalentLabeling(labelImage, referenceImage):
    x = labelImage
    y = referenceImage
    assert np.all(x.shape == y.shape),\
        "Shapes do not agree ({} vs {})".format(x.shape, y.shape)

    # identify labels used in x
    labels = set(x.flat)
    for label in labels:
        if label == 0:
            continue
        idx = np.where(x == label)
        refblock = y[idx]
        # check that labels are the same
        corner = [a[0] for a in idx]
        print("Inspecting region of size {} at {}".format(refblock.size, corner))

        assert np.all(refblock == refblock[0]),\
            "Uniformly labeled region at coordinates {} has more than one label in the reference image".format(corner)
        # check that nothing else is labeled with this label
        m = refblock.size
        n = (y == refblock[0]).sum()
        assert m == n, "There are more pixels with (reference-)label {} than pixels with label {}.".format(refblock[0], label)

    assert len(labels) == len(set(y.flat)), "The number of labels does not agree, perhaps some region was missed"

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

from itertools import imap, ifilter

import numpy as np
import vigra

from lazyflow.drtile import drtile as drtilecpp
from lazyflow.utility.fastWhere import fastWhere


def _drtile_old(bool_array):
    tileWeights = fastWhere(bool_array, 1, 128**3, np.uint32)
    return drtilecpp.test_DRTILE(tileWeights, 128**3)


def _drtile_new(bool_array):
    """
    tile the non-zero parts of a bool array into rectangles
 
    Returns an iterator that contains rois as concatenated start and
    stop vectors (for a 3-dim array, roi.shape == 6).

    Note: If you want to keep the tiles for more than one iteration, use
    this function as

    >>> tile_rois = list(drtile(bool_array))
    """
    x = bool_array

    if x.ndim == 1:
        return _drtile_1d(x)

    rois = []

    last = _drtile_new(x[..., 0])
    last_dict = dict((k, 0) for k in last)
    for i in range(1, x.shape[-1]):
        # compute tiling in hyperplane
        current = _drtile_new(x[..., i])
        # check which tiles are present in last hyperplane
        current_dict = dict(
            (k, (i if k not in last_dict else last_dict[k]))
            for k in current)

        # finalize tilings that are not in the current hyperplane
        final = ifilter(lambda k: k not in current_dict, last_dict)
        expanded = imap(lambda k: _expand_roi(k, last_dict[k], i), final)
        rois.extend(expanded)

        last_dict = current_dict

    # add the remaining tiles
    expanded = imap(lambda k: _expand_roi(k, last_dict[k], x.shape[-1]),
                    last_dict)
    rois.extend(expanded)
    return rois

def _expand_roi(r, a, b):
    n = len(r)
    h = n//2
    s = list(r)
    s.insert(h, a)
    s.append(b)
    return tuple(s)


def _drtile_1d(x):
    n = len(x)
    current = None
    for i in range(n):
        if current is None:
            # looking for a start
            if x[i]:
                current = i
        else:
            # looking for stop
            if not x[i]:
                yield ((current, i))
                current = None
    if current is not None:
        # we have an open region left
        yield ((current, n))



# export new style drtile
drtile = _drtile_new

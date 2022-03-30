# -*- coding: utf-8 -*-
# Copyright (C) 2006-2015  Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT (see COPYING file)

from __future__ import division
import numpy as np

__all__ = ['thin_3d']


def thin_3d(binimg, max_iter=-1):
    """
    skel = thin_3d(binimg)

    Skeletonisation by thinning

    Parameters
    ----------
    binimg : ndarray
        Binary input image
    max_iter : int, optional
        Maximum number of iterations (set to a negative number, the default, to
        run full skeletonization)

    Returns
    -------
    skel : Skeletonised version of `binimg`
    """

    from .bbox import bbox
    from ._thin_3d import thin_3d as _thin_3d

    res = np.zeros_like(binimg)
    min0,max0,min1,max1,min2,max2 = bbox(binimg)
    r,c,d = (max0-min0, max1-min1, max2-min2)

    image_exp = np.zeros((r+2, c+2, d+2), bool)
    image_exp[1:r+1, 1:c+1, 1:d+1] = binimg[min0:max0,min1:max1,min2:max2]
    imagebuf = np.empty((r+2,c+2,d+2), bool)

    _thin_3d(image_exp, imagebuf, int(max_iter))
    res[min0:max0, min1:max1, min2:max2] = image_exp[1:r+1, 1:c+1, 1:d+1]
    
    return res
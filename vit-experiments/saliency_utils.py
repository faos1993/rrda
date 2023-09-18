import numpy as np
from skimage.filters import gaussian

#source code based on https://github.com/josephdviviano/saliency-red-herring/blob/d697f87068bf7576e191e709fee6ec4306242165/activmask/results.py


SIGMA = 1


def _get_bin_loc_scores(salience_map, segs):
    segs = segs.cpu().numpy().astype(np.bool)
    salience_map = salience_map.astype(np.bool)
    
    locs_bin = np.zeros(salience_map.shape).astype(np.bool)
    idx = salience_map > 0
    locs_bin[idx] = 1
    locs = locs_bin
    
    EPS = 10e-16

    iou = (segs & locs).sum() / ((segs | locs).sum() + EPS)
    iop = (segs & locs).sum() / (locs.sum() + EPS)
    iot = (segs & locs).sum() / (segs.sum() + EPS)

    return {'iou':iou, 'iop':iop, 'iot':iot}


def threshold(x, percentile):
    return x * (x > np.percentile(x, percentile))


def clean_saliency(saliency, percentile=50.0, absoloute=True, blur=True):

    if absoloute:
        saliency = np.abs(saliency)

    if blur:
        saliency = gaussian(saliency,
                            mode='constant',
                            sigma=(SIGMA, SIGMA),
                            truncate=3.5,
                            preserve_range=True)

    if percentile > 0:
        saliency = threshold(saliency, percentile)

    return saliency
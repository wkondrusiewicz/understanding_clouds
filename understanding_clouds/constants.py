import numpy as np

NO_MASK_PROVIDED = 'no mask provided'
BACKGROUND_CLASSNAME = 'empty mask'

LABELS_MAPPING = {BACKGROUND_CLASSNAME: 0,
                  'Fish': 1,
                  'Flower': 2,
                  'Gravel': 3,
                  'Sugar': 4}

REVERSED_LABELS_MAPPING = {v: k for k, v in LABELS_MAPPING.items()}

LABELS_TO_COLORS = {BACKGROUND_CLASSNAME: [0, 0, 0],
                    'Fish': [0, 255, 0],
                    'Flower': [0, 0, 255],
                    'Gravel': [255, 0, 0],
                    'Sugar': [0, 255, 255]}


def colorize_mask(mask, label):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = LABELS_TO_COLORS[label]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

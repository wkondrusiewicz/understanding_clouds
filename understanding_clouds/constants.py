NO_MASK_PROVIDED = 'no mask provided'
BACKGROUND_CLASSNAME = 'empty mask'

LABELS_MAPPING = {BACKGROUND_CLASSNAME: 0,
                  'Fish': 1,
                  'Flower': 2,
                  'Gravel': 3,
                  'Sugar': 4}

REVERSED_LABELS_MAPPING = {v: k for k, v in LABELS_MAPPING.items()}

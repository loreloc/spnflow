import numpy as np
from spnflow.structure.leaf import LeafType


def get_data_domains(data, distributions):
    assert data is not None
    assert distributions is not None
    domains = []
    for i, d in enumerate(distributions):
        col = data[:, i]
        if d.LEAF_TYPE == LeafType.DISCRETE:
            domains.append(np.max(col).astype(int) + 1)
        elif d.LEAF_TYPE == LeafType.CONTINUOUS:
            domains.append([np.min(col), np.max(col)])
        else:
            raise NotImplementedError("Domain for leaf type " + d.LEAF_TYPE.__name__ + " not implemented")
    return domains

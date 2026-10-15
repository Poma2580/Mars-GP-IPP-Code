from copy import deepcopy
from mars_ipp.utils.config import merge


def expand(config):
    base = deepcopy(config)
    variants = base.pop('variants', None)
    sweep = base.pop('sweep', None)
    seeds = base.pop('seeds', [base['seed']])
    entries = []
    if variants:
        entries = [(name, merge(base, patch)) for name, patch in variants.items()]
    elif sweep:
        if len(sweep) != 1:
            raise ValueError('One sweep axis per experiment')
        path, values = next(iter(sweep.items()))
        section, key = path.split('.')
        entries = [(f'{key}_{value}', merge(base, {section: {key: value}})) for value in values]
    else:
        entries = [('', base)]
    for name, entry in entries:
        if name:
            entry['experiment'] += '_' + name
        for seed in seeds:
            yield merge(entry, {'seed': seed})

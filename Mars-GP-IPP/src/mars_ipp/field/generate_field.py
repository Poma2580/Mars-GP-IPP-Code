"""Create matched offline/online variants, never overwrite original field models."""
import hashlib
import json
import pickle
import GPy
from .dataset import load_dataset
from .field_model import FieldModel
from mars_ipp.gp.kernels import make_kernel
from mars_ipp.utils.config import resolve_path
from mars_ipp.utils.logger import write_json


def prepare_fields(config):
    spec = config['field']
    if spec['variant'] == 'original':
        return {name: FieldModel.load(resolve_path(path)) for name, path in spec['models'].items()}
    if spec['variant'] not in ('matern', 'lengthscale'):
        raise ValueError('Unknown field variant')
    workbook = resolve_path(spec['workbook'])
    provenance = {'variant': spec['variant'], 'kernel': spec['kernel'], 'factor': spec['lengthscale_factor'],
                  'workbook_sha256': hashlib.sha256(workbook.read_bytes()).hexdigest(),
                  'source_hashes': {name: hashlib.sha256(resolve_path(path).read_bytes()).hexdigest() for name, path in spec['models'].items()}}
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:16]
    directory = resolve_path('data/models/generated') / key
    directory.mkdir(parents=True, exist_ok=True)
    fields = {}
    for scenario in spec['models']:
        path = directory / f'{scenario}.pkl'
        if not path.exists():
            x, y = load_dataset(workbook, scenario)
            original = FieldModel.load(resolve_path(spec['models'][scenario]))
            if spec['variant'] == 'lengthscale':
                params = dict(original.parameters)
                params['lengthscale'] *= spec['lengthscale_factor']
                if params['lengthscale'] <= 0:
                    raise ValueError('Lengthscale must be positive')
                model = GPy.models.GPRegression(x, y, make_kernel(params), noise_var=params['noise'])
                model.kern.fix()
                model.Gaussian_noise.variance.fix()
            else:
                if spec['kernel']['type'] != 'matern':
                    raise ValueError('Matérn variant requires Matérn kernel')
                model = GPy.models.GPRegression(x, y, make_kernel(spec['kernel']))
                model.optimize(messages=False)
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as target:
                pickle.dump(model, target, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(path)
        fields[scenario] = FieldModel.load(path)
    write_json(directory / 'metadata.json', {**provenance, 'parameters': {name: field.parameters for name, field in fields.items()}})
    return fields

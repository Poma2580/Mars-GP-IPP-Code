import GPy


def make_kernel(spec):
    name = spec['type'].lower()
    if name == 'rbf':
        cls = GPy.kern.RBF
    elif name == 'matern' and spec.get('nu', 1.5) in (1.5, 2.5):
        cls = GPy.kern.Matern32 if spec.get('nu', 1.5) == 1.5 else GPy.kern.Matern52
    else:
        raise ValueError(f'Unsupported kernel: {spec}')
    return cls(input_dim=2, variance=spec['variance'], lengthscale=spec['lengthscale'])


def model_parameters(model):
    name = type(model.kern).__name__
    types = {'RBF': ('rbf', None), 'Matern32': ('matern', 1.5), 'Matern52': ('matern', 2.5)}
    if name not in types or model.Y.shape[1] != 3 or model.kern.input_dim != 2:
        raise ValueError('Expected one shared RBF/Matérn kernel with three outputs')
    kernel, nu = types[name]
    return {'type': kernel, 'nu': nu, 'variance': float(model.kern.variance[0]),
            'lengthscale': float(model.kern.lengthscale[0]), 'noise': float(model.Gaussian_noise.variance[0])}

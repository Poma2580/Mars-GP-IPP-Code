import pytest
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.utils.config import merge, ROOT, load_config


@pytest.mark.parametrize('factor', [.5, 1., 2., 3.])
def test_lengthscale_is_locked(config, factor):
    original = prepare_fields(config)
    fields = prepare_fields(merge(config, {'field': {'variant': 'lengthscale', 'lengthscale_factor': factor}}))
    for name, field in fields.items():
        assert field.parameters['lengthscale'] == pytest.approx(original[name].parameters['lengthscale'] * factor)
        assert field.parameters['variance'] == original[name].parameters['variance']
        assert field.parameters['noise'] == original[name].parameters['noise']
        assert field.model.kern.is_fixed


@pytest.mark.parametrize('nu', [1.5, 2.5])
def test_matern_matched_online_parameters(config, nu):
    from mars_ipp.gp.estimator import OnlineGP
    fields = prepare_fields(merge(config, {'field': {'variant': 'matern', 'kernel': {'type': 'matern', 'nu': nu}}}))
    for field in fields.values():
        gp = OnlineGP(field.parameters)
        gp.add([0,0], field.truth([[0,0]])[0])
        assert gp.parameters == field.parameters
        assert gp.parameters['nu'] == nu
        assert gp.model.kern.is_fixed


def test_all_yaml_configs_valid():
    for path in (ROOT / 'configs').rglob('*.yaml'):
        cfg = load_config(path)
        assert cfg['env']['evaluation_size'] == 31


@pytest.mark.parametrize('variant', ['lengthscale', 'matern'])
def test_generated_cache_creation_and_reuse(config, tmp_path, monkeypatch, variant):
    import hashlib
    import mars_ipp.field.generate_field as generation
    resolve = generation.resolve_path
    monkeypatch.setattr(generation, 'resolve_path',
                        lambda path: tmp_path / 'generated' if str(path) == 'data/models/generated' else resolve(path))
    originals = {name: hashlib.sha256(resolve(path).read_bytes()).hexdigest()
                 for name, path in config['field']['models'].items()}
    patch = {'variant': variant, 'lengthscale_factor': 1.0}
    if variant == 'matern':
        patch['kernel'] = {'type': 'matern', 'nu': 1.5}
    settings = merge(config, {'field': patch})
    first = prepare_fields(settings)
    assert len(list((tmp_path / 'generated').rglob('*.pkl'))) == 2
    assert list((tmp_path / 'generated').rglob('metadata.json'))

    def unexpected_load(*args):
        raise AssertionError('Cached fields must not be rebuilt from the workbook')

    monkeypatch.setattr(generation, 'load_dataset', unexpected_load)
    second = prepare_fields(settings)
    for name in first:
        assert first[name].sha256 == second[name].sha256
        assert first[name].source == second[name].source
        assert str(tmp_path) in first[name].source
    assert originals == {name: hashlib.sha256(resolve(path).read_bytes()).hexdigest()
                         for name, path in config['field']['models'].items()}

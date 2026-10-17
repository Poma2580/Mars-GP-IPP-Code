"""Map-only evaluation must not enter historical recovery or overwrite results."""
import importlib

import pytest

from mars_ipp.utils.config import ROOT


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    return importlib.import_module('complete_v3_1_recovery')


def test_selected_test1_leaves_existing_results_and_report(runner, monkeypatch, tmp_path):
    old = tmp_path / 'test2'
    old.mkdir()
    (old / 'saved.txt').write_text('existing results')
    report = tmp_path / 'report.md'
    report.write_text('historical report')
    cfg = {'v3': {'original_source': 'unused.pkl', 'sha256': 'digest'}}
    monkeypatch.setattr(runner, 'read_checkpoint', lambda _: {'config': cfg})
    monkeypatch.setattr(runner, 'checkpoint_for', lambda *_: None)
    monkeypatch.setattr(runner, 'sha256', lambda _: 'digest')

    def forbidden(*_):
        pytest.fail('Map-only evaluation entered historical recovery')

    monkeypatch.setattr(runner, 'historical_check', forbidden)
    monkeypatch.setattr(runner, 'run_recovery', forbidden)
    selected = []

    def evaluate_map(scene, output, *args):
        selected.append((scene, args[-1]))
        (output / scene).mkdir()

    monkeypatch.setattr(runner, 'evaluate_map', evaluate_map)
    runner.run(('test1',), tmp_path, (1000,))
    assert selected == [('test1', (1000,))]
    assert (tmp_path / 'test1/summary/input_hashes.json').is_file()
    assert (old / 'saved.txt').read_text() == 'existing results'
    assert report.read_text() == 'historical report'


def test_existing_selected_directory_rejected_before_loading_models(runner, monkeypatch, tmp_path):
    (tmp_path / 'test1').mkdir()
    monkeypatch.setattr(runner, 'read_checkpoint', lambda _: pytest.fail('Loaded model before checking output'))
    with pytest.raises(FileExistsError):
        runner.run(('test1',), tmp_path)


@pytest.mark.parametrize('maps,seeds', [(('test1', 'test1'), (1000,)),
                                      (('test6',), (1000,)),
                                      (('test1',), ()),
                                      (('test1',), (1000, 1000))])
def test_invalid_selection_creates_no_output(runner, tmp_path, maps, seeds):
    output = tmp_path / 'new'
    with pytest.raises(ValueError):
        runner.run(maps, output, seeds)
    assert not output.exists()

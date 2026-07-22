import json

import pytest

import localizr.cli as cli_module


class _FakeResult:
    def __init__(self, verdict="on_target"):
        self.tic_id = None
        self.kic_id = 4281068
        self.mission = "Kepler"
        self.verdict = verdict
        self.verdict_reason = f"fake reason: {verdict}"
        self.centroid_offset_arcsec = 1.23
        self.centroid_offset_sigma = 5.0 if verdict == "off_target_blend" else 0.5
        self.target_position = {"ra": 1.0, "dec": 2.0}
        self.difference_image_centroid = {"ra": 1.0001, "dec": 2.0001}
        self.nearby_gaia_sources = [
            {"source_id": 42, "ra": 1.0, "dec": 2.0, "sep_arcsec": 1.0, "gmag": 12.0}
        ]

    def save_plot(self, path):
        with open(path, "wb") as f:
            f.write(b"fake-png")
        return path

    def to_json(self, path=None):
        text = json.dumps({"verdict": self.verdict})
        if path is not None:
            with open(path, "w") as f:
                f.write(text)
        return text


def test_main_requires_a_command():
    with pytest.raises(SystemExit):
        cli_module.main([])


def test_main_missing_required_ephemeris_args_exits_nonzero():
    with pytest.raises(SystemExit):
        cli_module.main(["localize", "--tic-id", "1"])


def test_main_on_target_exit_code(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module, "localize", lambda **kw: _FakeResult("on_target"))
    code = cli_module.main(
        ["localize", "--kic-id", "4281068", "--period", "1.0", "--epoch", "0.0", "--duration-hours", "1.0"]
    )
    assert code == 0


def test_main_off_target_blend_exit_code(monkeypatch):
    monkeypatch.setattr(cli_module, "localize", lambda **kw: _FakeResult("off_target_blend"))
    code = cli_module.main(
        ["localize", "--kic-id", "4281068", "--period", "1.0", "--epoch", "0.0", "--duration-hours", "1.0"]
    )
    assert code == 1


def test_main_json_output(monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "localize", lambda **kw: _FakeResult("on_target"))
    cli_module.main(
        ["localize", "--kic-id", "4281068", "--period", "1.0", "--epoch", "0.0",
         "--duration-hours", "1.0", "--json"]
    )
    out = capsys.readouterr().out
    assert json.loads(out) == {"verdict": "on_target"}


def test_main_plot_written(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module, "localize", lambda **kw: _FakeResult("on_target"))
    plot_path = tmp_path / "out.png"
    cli_module.main(
        ["localize", "--kic-id", "4281068", "--period", "1.0", "--epoch", "0.0",
         "--duration-hours", "1.0", "--plot", "--plot-path", str(plot_path)]
    )
    assert plot_path.exists()


def test_main_reports_resolution_error(monkeypatch, capsys):
    from localizr.tpf import TpfResolutionError

    def raiser(**kw):
        raise TpfResolutionError("no TPF found")

    monkeypatch.setattr(cli_module, "localize", raiser)
    code = cli_module.main(
        ["localize", "--kic-id", "999999999", "--period", "1.0", "--epoch", "0.0", "--duration-hours", "1.0"]
    )
    assert code == 2
    assert "no TPF found" in capsys.readouterr().out

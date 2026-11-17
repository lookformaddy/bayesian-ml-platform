from bayesian.cli import main


def test_cli_demo(capsys):
    assert main(["demo"]) == 0
    assert "Promotion" in capsys.readouterr().out


def test_cli_train(capsys):
    assert main(["train", "--version", "v1"]) == 0
    out = capsys.readouterr().out
    assert '"promoted": true' in out


def test_cli_predict(capsys):
    assert main(["predict", "0.1", "-0.2", "0.3", "0.0", "0.5", "-0.1"]) == 0
    out = capsys.readouterr().out
    assert '"action"' in out


def test_cli_bandit(capsys):
    assert main(["bandit"]) == 0
    assert "thompson_sampling" in capsys.readouterr().out


def test_cli_drift(capsys):
    assert main(["drift"]) == 0
    assert "recalibrate_and_refit" in capsys.readouterr().out

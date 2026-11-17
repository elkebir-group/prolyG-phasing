"""The command-line argument parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

import prolyg_phasing.cli as cli

_BUILD_FLANKS = ["build-flanks", "--bam", "n.bam", "--bed", "p.bed", "--out-prefix", "out/n"]


def test_build_flanks_workers_defaults_to_none():
    args = cli.build_parser().parse_args(_BUILD_FLANKS)
    assert args.workers is None
    assert (args.bam, args.bed, args.out_prefix) == (Path("n.bam"), Path("p.bed"),
                                                     Path("out/n"))


def test_build_flanks_takes_a_worker_count():
    assert cli.build_parser().parse_args([*_BUILD_FLANKS, "--workers", "6"]).workers == 6


@pytest.mark.parametrize("value", ["0", "-2", "two"])
def test_build_flanks_rejects_a_worker_count_below_one(value):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([*_BUILD_FLANKS, "--workers", value])


@pytest.mark.parametrize(("argv", "want"), [([], 5), (["--workers", "3"], 3)])
def test_build_flanks_uses_the_worker_count(monkeypatch, argv, want):
    """Without --workers the count comes from worker_count(); with it, from the flag."""
    seen = {}
    monkeypatch.setattr(cli, "worker_count", lambda: 5)

    def fake_build_flanks(bam, bed, *, n_workers, sample):
        seen["n_workers"] = n_workers
        return type("Result", (), {"sample": "S", "records": {}})()

    monkeypatch.setattr(cli, "build_flanks", fake_build_flanks)
    monkeypatch.setattr(cli, "write_flank_table", lambda result, prefix: Path(prefix))
    cli.main([*_BUILD_FLANKS, *argv])
    assert seen["n_workers"] == want

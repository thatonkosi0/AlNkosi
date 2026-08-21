"""CLI launcher: the non-loopback host-binding guard."""

from __future__ import annotations

import pytest

from alglory.__main__ import _check_host


def test_loopback_hosts_allowed_without_flag():
    _check_host("127.0.0.1", allow_remote=False)
    _check_host("localhost", allow_remote=False)


def test_non_loopback_host_refused_without_allow_remote():
    with pytest.raises(SystemExit):
        _check_host("0.0.0.0", allow_remote=False)


def test_non_loopback_host_allowed_with_allow_remote():
    _check_host("0.0.0.0", allow_remote=True)  # must not raise

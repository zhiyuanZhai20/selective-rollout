"""
File-backed, flock-protected line queue. Multiple workers can atomically pop.

Format: plain text, one item per line. `pop()` removes the first remaining line and
returns it (or None if empty). `remaining()` peeks at count.

Locking uses fcntl.flock with LOCK_EX — safe across processes on the same host.
"""
from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager


@contextmanager
def _locked(path: str):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def init(path: str, items: list[str]) -> None:
    with _locked(path) as fd:
        os.ftruncate(fd, 0)
        os.write(fd, ("\n".join(items) + ("\n" if items else "")).encode())


def pop(path: str) -> str | None:
    with _locked(path) as fd:
        os.lseek(fd, 0, os.SEEK_SET)
        data = os.read(fd, os.fstat(fd).st_size).decode()
        lines = data.splitlines()
        if not lines:
            return None
        first, rest = lines[0], lines[1:]
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        if rest:
            os.write(fd, ("\n".join(rest) + "\n").encode())
        return first


def remaining(path: str) -> int:
    try:
        with _locked(path) as fd:
            os.lseek(fd, 0, os.SEEK_SET)
            data = os.read(fd, os.fstat(fd).st_size).decode()
            return sum(1 for ln in data.splitlines() if ln.strip())
    except FileNotFoundError:
        return 0


@contextmanager
def append_lock(path: str):
    """Lock an output file for the duration of a single append."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def append_line(path: str, line: str) -> None:
    with append_lock(path) as fd:
        if not line.endswith("\n"):
            line = line + "\n"
        os.write(fd, line.encode())

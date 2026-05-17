"""Thin async wrapper around `git worktree` for workspace bootstrap.

App-layer concern. Not part of agent-webkit. Lives next to the workspace
store and is the only place we shell out to `git` for worktree mechanics.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class WorktreeError(Exception):
    code: str
    message: str
    detail: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._/\-]+$")


def _validate_branch(branch: str) -> None:
    if not branch or not _SAFE_NAME_RE.match(branch):
        raise WorktreeError(
            code="invalid_branch",
            message=f"Invalid branch name: {branch!r}",
        )


async def _run(*args: str, cwd: Optional[str] = None) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode or 0, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")


async def is_git_repo(path: str) -> bool:
    """True if `path` is inside a git working tree."""
    p = Path(path).expanduser()
    if not p.is_dir():
        return False
    code, _out, _err = await _run("git", "-C", str(p), "rev-parse", "--is-inside-work-tree")
    return code == 0


async def current_branch(repo_path: str) -> Optional[str]:
    code, out, _err = await _run("git", "-C", str(Path(repo_path).expanduser()), "rev-parse", "--abbrev-ref", "HEAD")
    if code != 0:
        return None
    branch = out.strip()
    return branch or None


async def branch_exists(repo_path: str, branch: str) -> bool:
    """True if the given branch already exists in the repo (local or remote)."""
    _validate_branch(branch)
    repo = str(Path(repo_path).expanduser())
    # Local
    code, _o, _e = await _run("git", "-C", repo, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}")
    if code == 0:
        return True
    # Remote (any remote)
    code, out, _e = await _run("git", "-C", repo, "for-each-ref", "--format=%(refname)", "refs/remotes/")
    if code == 0:
        for line in out.splitlines():
            if line.endswith(f"/{branch}"):
                return True
    return False


async def has_uncommitted_changes(repo_path: str) -> bool:
    code, out, _err = await _run("git", "-C", str(Path(repo_path).expanduser()), "status", "--porcelain")
    if code != 0:
        return False
    return bool(out.strip())


async def add_worktree(
    repo_path: str,
    worktree_path: str,
    branch: str,
    *,
    base: Optional[str] = None,
) -> None:
    """Create a new worktree at `worktree_path` on a new branch `branch`.

    If `branch` already exists, the worktree is checked out on it (no -b).
    If `base` is provided, the new branch is created from that ref;
    otherwise from the repo's current HEAD.

    Raises WorktreeError on any git failure.
    """
    _validate_branch(branch)
    if base is not None:
        _validate_branch(base)
    repo = str(Path(repo_path).expanduser())
    worktree = str(Path(worktree_path).expanduser())
    if not await is_git_repo(repo):
        raise WorktreeError(
            code="not_a_repo",
            message=f"Source path is not a git repository: {repo}",
        )
    if Path(worktree).exists():
        raise WorktreeError(
            code="worktree_path_exists",
            message=f"Worktree target already exists: {worktree}",
        )

    Path(worktree).parent.mkdir(parents=True, exist_ok=True)

    if await branch_exists(repo, branch):
        # Checkout existing branch into a new worktree.
        args = ["git", "-C", repo, "worktree", "add", worktree, branch]
    else:
        # Create a new branch in the new worktree.
        args = ["git", "-C", repo, "worktree", "add", "-b", branch, worktree]
        if base is not None:
            args.append(base)

    code, _out, err = await _run(*args)
    if code != 0:
        raise WorktreeError(
            code="git_worktree_add_failed",
            message=f"git worktree add failed for {repo} -> {worktree}",
            detail=err.strip() or None,
        )


async def remove_worktree(repo_path: str, worktree_path: str, *, force: bool = False) -> None:
    """Tear down a worktree. `force` skips uncommitted-changes safety."""
    repo = str(Path(repo_path).expanduser())
    worktree = str(Path(worktree_path).expanduser())
    args = ["git", "-C", repo, "worktree", "remove", worktree]
    if force:
        args.append("--force")
    code, _out, err = await _run(*args)
    if code != 0:
        # If git refuses, fall back to brute removal so the workspace store
        # doesn't get stuck holding a dangling reference. Caller's choice
        # to invoke this typically already gated on confirmation.
        if force and Path(worktree).exists():
            shutil.rmtree(worktree, ignore_errors=True)
            return
        raise WorktreeError(
            code="git_worktree_remove_failed",
            message=f"git worktree remove failed for {worktree}",
            detail=err.strip() or None,
        )

"""Incremental GitHub publishing for one layout-fix training session.

Each session pushes to its own new branch, so parallel notebooks never conflict. The branch holds only results,
in an orphan history, under the same path the session uses locally (output/layout_fix_study/layout-fix-v1/<session>),
so checking the files out of the branch puts them where confirm_block_layout.py expects them. Every sync makes one
commit and a fast-forward push, so completed model pairs survive a notebook timeout.

Checkpoints are published weights-only (model state, settings and step counters; no optimizer state), about 6.5 MB
each instead of about 20 MB. That is all evaluation needs; training cannot be resumed from them.
"""
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

from artifacts import atomic_write_json

MAX_FILE_BYTES = 90 * 1024 * 1024


class SessionPublisher:
    def __init__(self, session, branch, remote=None):
        from kaggle_session import remote_url
        self.session = Path(session)
        self.branch = branch
        self.remote = remote or remote_url('origin')
        subprocess.run(['git', 'check-ref-format', 'refs/heads/' + branch], check=True)
        self.staging = Path(tempfile.mkdtemp(prefix='layout-fix-publish-'))
        self.target = self.staging / 'output' / 'layout_fix_study' / 'layout-fix-v1' / self.session.name
        self.initialized = False
        self.pushes = 0

    def _git(self, *arguments, env=None):
        subprocess.run(['git', *arguments], cwd=self.staging, env=env, check=True, timeout=600)

    def _copy(self):
        files = []
        for source in sorted(self.session.rglob('*')):
            if not source.is_file() or source.is_symlink() or source.suffix == '.tmp':
                continue
            relative = source.relative_to(self.session)
            destination = self.target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == '.pt':
                if not destination.exists() or destination.stat().st_mtime < source.stat().st_mtime:
                    saved = torch.load(source, map_location='cpu', weights_only=False)
                    torch.save({key: saved[key] for key in ('model', 'settings', 'completed_steps', 'completed_epochs')
                                if key in saved}, destination)
            else:
                shutil.copy2(source, destination)
            size = destination.stat().st_size
            if size >= MAX_FILE_BYTES:
                raise ValueError(f'Artifact too large for GitHub: {relative}')
            files.append(dict(path=relative.as_posix(), bytes=size,
                              sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                              weights_only_checkpoint=source.suffix == '.pt'))
        atomic_write_json(self.target / 'publish_manifest.json', dict(
            branch=self.branch, session=self.session.name, files=files,
            note='Checkpoints are weights-only copies (no optimizer state); local session files are unchanged.'))

    def sync(self, message):
        from kaggle_session import git_environment
        self._copy()
        with git_environment() as env:
            if not self.initialized:
                self._git('init', '-b', self.branch)
                self._git('config', 'user.name', 'Research results')
                self._git('config', 'user.email', 'research-results@users.noreply.github.com')
                self.initialized = True
            self._git('add', '-A', '--', 'output')
            if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=self.staging).returncode == 0:
                return False
            self._git('commit', '-q', '-m', message)
            self._git('push', self.remote, f'HEAD:refs/heads/{self.branch}', env=env)
        self.pushes += 1
        return True

    def close(self):
        shutil.rmtree(self.staging, ignore_errors=True)


def default_branch(protocol, arms, session_name, variants=('scalar', 'channel-wise')):
    suffix = '' if len(variants) == 2 else f'-{variants[0]}'
    return f'research/layout-fix/{protocol}/{"+".join(arms)}{suffix}/{session_name}'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Publish (or re-publish) an existing session folder to its branch.')
    parser.add_argument('session', type=Path)
    parser.add_argument('--branch', required=True)
    parser.add_argument('--remote')
    args = parser.parse_args()
    publisher = SessionPublisher(args.session, args.branch, args.remote)
    try:
        publisher.sync(f'Record layout-fix session {args.session.name}')
        print('Published results branch:', args.branch, flush=True)
    finally:
        publisher.close()

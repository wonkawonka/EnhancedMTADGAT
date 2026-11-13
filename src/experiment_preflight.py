"""Shared fail-closed launch checks; source review remains separate evidence.

Only callers of this module are gated. This does not retrofit legacy runners.
Evidence is tied to file content, never accepted solely because it says passed.
"""
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_evidence(path, required_files):
    evidence = json.loads(Path(path).read_text())
    if evidence.get('status') != 'passed':
        raise ValueError(f'Unpassed evidence: {path}')
    hashes = evidence.get('files', {})
    if not required_files:
        raise ValueError('Evidence needs an explicit dependency list')
    for name in required_files:
        if name not in hashes or sha256(name) != hashes[name]:
            raise ValueError(f'Missing or stale evidence: {name}')
    return evidence


class Preflight:
    def __init__(self, output, plan_path, scope):
        self.path = Path(output)/'preflight.json'
        self.record = dict(scope=scope, plan_sha256=sha256(plan_path), checks={})
        self.flush()

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.record, indent=2, ensure_ascii=False))
        temporary.replace(self.path)

    def check(self, name, condition, detail):
        self.record['checks'][name] = dict(status='passed' if condition else 'failed', detail=detail)
        self.flush()
        if not condition:
            raise ValueError(f'Preflight failed: {name}: {detail}')

    def not_checked(self, name, detail):
        self.record['checks'][name] = dict(status='not_checked', detail=detail)
        self.flush()

    def require(self, names):
        missing = [n for n in names if self.record['checks'].get(n, {}).get('status') != 'passed']
        if missing:
            raise ValueError(f'Required verification not passed: {missing}')

    def formal_gate(self, plan, effective, required_checks, evidence):
        self.check('formal_plan_approved', plan.get('approved') is True and plan.get('runnable') is True,
                   'Formal runs require an approved runnable plan')
        self.check('resolved_config', effective == plan.get('effective_config'),
                   {'actual': effective, 'expected': plan.get('effective_config')})
        self.check('evidence_present', bool(evidence), 'Source/numerical/data evidence required')
        for name, spec in evidence.items():
            try:
                verify_evidence(spec['path'], spec['required_files'])
            except (ValueError, KeyError, OSError) as error:
                self.check(name, False, str(error))
            self.check(name, True, spec)
        self.require(required_checks)

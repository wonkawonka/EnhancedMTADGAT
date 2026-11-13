import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.compact_battery_windows import CompactSnippetWindows
from src.experiment_preflight import Preflight, sha256, verify_evidence


class CompactPreflightTests(unittest.TestCase):
    def test_compact_order_and_boundaries(self):
        for span,stride in [(10,1),(101,1),(100,100),(105,1),(128,128)]:
            with self.subTest(span=span):
                raw=np.arange(4*128*7).reshape(4,128,7)
                ds=CompactSnippetWindows([3,1],128,span,stride)
                starts=list(range(0,128-span+1,stride))
                if starts[-1]!=128-span:starts.append(128-span)
                reference=np.stack([raw[row,start:start+span] for row in [3,1] for start in starts])
                np.testing.assert_array_equal(ds.gather(raw,np.arange(len(ds))),reference)
                if span==101:np.testing.assert_array_equal(reference[0,100],raw[3,100])
                with self.assertRaises(IndexError):ds.decode([len(ds)])

    def test_stale_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.py';source.write_text('version1')
            proof=root/'proof.json'
            proof.write_text(json.dumps({'status':'passed','files':{str(source):sha256(source)}}))
            verify_evidence(proof,[str(source)])
            source.write_text('version2')
            with self.assertRaises(ValueError):verify_evidence(proof,[str(source)])

    def test_unchecked_and_draft_cannot_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);plan=root/'plan.json';plan.write_text('{}')
            audit=Preflight(root,plan,'test')
            audit.not_checked('native_score','pending')
            with self.assertRaises(ValueError):audit.require(['native_score'])
            with self.assertRaises(ValueError):audit.formal_gate({'approved':False},{},[],{})


if __name__=='__main__':unittest.main()

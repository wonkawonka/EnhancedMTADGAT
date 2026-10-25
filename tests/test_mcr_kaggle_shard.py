"""CPU checks for four-shard coverage, minimal inputs and exported symlinks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_mcr_priority5_kaggle import dependencies, PLAN, inputs_for
from src.runners.run_mcr_kaggle_shard import regularize_context_markers, setup_runtime


class KaggleShardTests(unittest.TestCase):
    def test_four_shards_cover_seven_units_once(self):
        plan = json.loads(Path(PLAN).read_text())
        units = [u for shard in plan['shards'].values() for u in shard['units']]
        self.assertEqual(len(plan['shards']), 4)
        self.assertEqual(len(units), len(set(units)))
        self.assertEqual(set(units), {'LFP','NCM',*(f'Brand3_fold{i}' for i in range(5))})
        self.assertEqual(11*len(units), 77)
        self.assertEqual([3407,3408,3409], plan['seeds'])
        files = dependencies(['src/runners/run_mcr_kaggle_shard.py'])
        self.assertIn('src/models/mcr_direct_context.py', files)
        self.assertIn('src/models/mscm_reconstruction.py', files)
        self.assertIn('scripts/evaluate_rcr.py', files)
        self.assertNotIn('scripts/rcr_kaggle_account.py', files)
        self.assertFalse(any('.kaggle' in f or f.startswith('runs/') for f in files))

    def test_incomplete_parent_is_never_packaged(self):
        plan = json.loads(Path(PLAN).read_text())
        with tempfile.TemporaryDirectory() as tmp, patch('scripts.prepare_mcr_priority5_kaggle.PROJECT', Path(tmp)):
            with self.assertRaisesRegex(FileNotFoundError, 'not fully scored'):
                inputs_for(plan, ['LFP'])

    def test_extra_seed_exports_do_not_duplicate_shared_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root/'model_two/NCM'
            canonical.mkdir(parents=True)
            (canonical/'conditions_complete.json').write_text('{"raw_target_context_max_error":0}')
            (canonical/'train_direct_context.npy').write_bytes(b'fixture')
            for seed in [3408,3409]:
                folder=root/f'relation_seed{seed}/model_two/NCM'
                folder.mkdir(parents=True)
                for name in ['conditions_complete.json','train_direct_context.npy']:
                    (folder/name).symlink_to(canonical/name)
            regularize_context_markers(root,'NCM')
            for seed in [3408,3409]:
                folder=root/f'relation_seed{seed}/model_two/NCM'
                self.assertTrue((folder/'conditions_complete.json').is_file())
                self.assertFalse((folder/'conditions_complete.json').is_symlink())
                self.assertFalse((folder/'train_direct_context.npy').exists())
            self.assertEqual((canonical/'train_direct_context.npy').read_bytes(),b'fixture')


if __name__ == '__main__':
    unittest.main()

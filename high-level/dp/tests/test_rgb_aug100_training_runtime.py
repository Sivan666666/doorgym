"""Regression tests for long TMPDIR IPC failure and live-but-stalled training."""
import argparse
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'train'))
from run_pull_rgb_aug100_pi05_0912_ps1 import TrainingWatch


class ProgressTests(unittest.TestCase):
    def test_finite_loss_and_split_lines(self):
        w=TrainingWatch(now=0)
        w.feed('INFO step:50 smpl:800 loss:',now=5)
        self.assertIsNone(w.latest)
        w.feed('0.123 grdn:1.0\n',now=10)
        self.assertEqual(w.latest['confirmed_steps'],50)
        self.assertEqual(w.latest['loss'],.123)
        w.feed('INFO step:100 smpl:1.6K loss:0.1\n',now=20)
        self.assertEqual(w.latest['confirmed_steps'],100)

    def test_worker_exception(self):
        w=TrainingWatch(now=0)
        w.feed('OSError: AF_UNIX path too ',now=1)
        with self.assertRaisesRegex(RuntimeError,'worker/runtime'):
            w.feed('long\n',now=2)

    def test_traceback(self):
        with self.assertRaises(RuntimeError):
            TrainingWatch(now=0).feed('Traceback (most recent call last):\n',now=1)

    def test_log_noise_does_not_mask_stall(self):
        w=TrainingWatch(now=0)
        w.feed('still alive\n',now=1801)
        with self.assertRaisesRegex(RuntimeError,'stalled'):w.check_stalled(now=1801)

    def test_nan_and_inf(self):
        for loss in ('nan','inf','-inf'):
            with self.assertRaisesRegex(RuntimeError,'Non-finite'):
                TrainingWatch(now=0).feed(f'step:50 loss:{loss}\n',now=5)


class IPCTests(unittest.TestCase):
    def test_worker_tensor_transfer(self):
        import socket
        import torch
        from torch.utils.data import DataLoader,TensorDataset
        # Same pathname layout that multiprocessing.resource_sharer uses.
        with tempfile.TemporaryDirectory(prefix='pymp-') as tmp:
            path=str(Path(tmp)/'listener-12345678')
            self.assertLess(len(path.encode()),108)
            with socket.socket(socket.AF_UNIX) as sock:sock.bind(path)
        expected=torch.arange(64).reshape(16,4)
        loader=DataLoader(TensorDataset(expected),batch_size=4,num_workers=4,timeout=60)
        actual=torch.cat([batch[0] for batch in loader])
        torch.testing.assert_close(actual,expected)
        print('IPC OK: four DataLoader workers, tensor transfer; TMPDIR='+tempfile.gettempdir(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ipc-only',action='store_true')
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(IPCTests) if args.ipc_only else unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

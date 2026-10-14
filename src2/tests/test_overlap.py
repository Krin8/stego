"""Process snapshots only: never launch, stop, or train any model."""
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from src2.train_segmentation import overlap_check


HEADER = 'PID PPID STATE ELAPSED COMMAND\n'
SELF = '28311 28000 S+ 00:01 python3 -m src2.train_segmentation\n'


class OverlapTests(unittest.TestCase):
    def check_snapshot(self, rows):
        snapshot = HEADER + SELF + rows
        with patch('src2.train_segmentation.os.getpid', return_value=28311), \
             patch('src2.train_segmentation.os.getppid', return_value=28000), \
             patch('src2.train_segmentation.subprocess.run', return_value=SimpleNamespace(stdout=snapshot)):
            return overlap_check()

    def test_own_caffeinate_child_is_not_another_training_job(self):
        for command in ('caffeinate', '/usr/bin/caffeinate'):
            snapshot = self.check_snapshot(f'28312 28311 S+ 00:01 {command} -i python3 -m src2.train_segmentation\n')
            self.assertIn('28312', snapshot)

    def test_own_caffeinate_parent_is_not_another_training_job(self):
        self.check_snapshot('28000 27000 S+ 00:01 caffeinate -i python3 -m src2.train_segmentation\n')

    def test_unrelated_caffeinate_still_blocks(self):
        with self.assertRaisesRegex(RuntimeError, '29001'):
            self.check_snapshot('29001 29000 S+ 00:05 caffeinate -i python3 -m src2.train_segmentation\n')

    def test_other_python_including_child_or_suspended_still_blocks(self):
        for ppid, state in ((28311, 'S+'), (29000, 'T')):
            with self.assertRaisesRegex(RuntimeError, '29002'):
                self.check_snapshot(f'29002 {ppid} {state} 00:05 python3 -m src2.train_segmentation\n')

    def test_ps_failure_still_stops(self):
        with patch('src2.train_segmentation.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'ps')):
            with self.assertRaises(subprocess.CalledProcessError): overlap_check()


if __name__ == '__main__': unittest.main()

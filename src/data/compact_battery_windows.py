"""Boundary-preserving windows over a shared fixed-length snippet pack."""
import numpy as np


class CompactSnippetWindows:
    def __init__(self, rows, length, span, stride):
        self.rows = np.asarray(rows, dtype=np.int64)
        if self.rows.ndim != 1 or span < 1 or stride < 1 or length < span:
            raise ValueError('Invalid snippet/window dimensions')
        self.span = span
        self.starts = np.arange(0, length-span+1, stride, dtype=np.int64)
        if self.starts[-1] != length-span:
            self.starts = np.append(self.starts, length-span)

    def __len__(self):
        return len(self.rows)*len(self.starts)

    def decode(self, positions):
        positions = np.asarray(positions, dtype=np.int64)
        if np.any(positions < 0) or np.any(positions >= len(self)):
            raise IndexError('Window position outside partition')
        return (self.rows[positions//len(self.starts)],
                self.starts[positions % len(self.starts)])

    def gather(self, raw, positions):
        rows, starts = self.decode(positions)
        return raw[rows[:, None], starts[:, None]+np.arange(self.span)[None, :], :]

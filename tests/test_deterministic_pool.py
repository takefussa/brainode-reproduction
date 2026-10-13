"""Check exact maximum pooling values and tie gradients against native CPU."""

import unittest
import torch
from deterministic_pool import DisjointMaxPool3D


class DisjointPoolTests(unittest.TestCase):
    def test_values_and_gradients_including_ties(self):
        torch.manual_seed(9)
        for value in (torch.randn(2, 3, 8, 6, 4), torch.zeros(2, 3, 8, 6, 4)):
            a, b = value.clone().requires_grad_(), value.clone().requires_grad_()
            reference, actual = torch.nn.functional.max_pool3d(a, 2), DisjointMaxPool3D()(b)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            weights = torch.randn_like(actual)
            (reference*weights).sum().backward(); (actual*weights).sum().backward()
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

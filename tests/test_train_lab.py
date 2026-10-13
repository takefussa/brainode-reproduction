import unittest

import torch
from torch import nn

from train_lab import predict_future


class ConditionVelocity(nn.Module):
    def forward(self,t,value,c):
        return torch.ones_like(value)*c[0]


class FuturePredictionTests(unittest.TestCase):
    def test_initial_c_only_with_all_replicate_targets_retained(self):
        # Future shapes are deliberately incompatible with the prediction.
        shapes=torch.tensor([[3.],[900.],[-900.],[10000.]])
        times=torch.tensor([0.,.25,.25,.5])
        targets,predicted=predict_future(ConditionVelocity(),shapes,times,torch.tensor(.4))
        self.assertEqual(targets.tolist(),[1,2,3])
        torch.testing.assert_close(predicted[:,0],torch.tensor([3.1,3.1,3.2]))
        altered=shapes.clone();altered[1:]+=100000
        _,second=predict_future(ConditionVelocity(),altered,times,torch.tensor(.4))
        torch.testing.assert_close(predicted,second)

    def test_same_time_replicates_do_not_become_future_targets(self):
        targets,predicted=predict_future(ConditionVelocity(),torch.zeros(2,3),
            torch.tensor([.2,.2]),torch.tensor(1.))
        self.assertEqual(len(targets),0)
        self.assertEqual(tuple(predicted.shape),(0,3))


if __name__=='__main__':
    unittest.main()

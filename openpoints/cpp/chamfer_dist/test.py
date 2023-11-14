import os
import sys
import torch
import unittest

from torch.autograd import gradcheck

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from openpoints.cpp.chamfer_dist import ChamferFunction


class ChamferDistanceTestCase(unittest.TestCase):
    def test_chamfer_dist(self):
        x = torch.rand(4, 64, 3).float()
        y = torch.rand(4, 128, 3).float()
        x.requires_grad = True
        y.requires_grad = True
        print(gradcheck(ChamferFunction.apply, [x.cuda(), y.cuda()]))


if __name__ == '__main__':
    x = torch.rand(32, 128, 3)
    y = torch.rand(32, 128, 3)
    # print()

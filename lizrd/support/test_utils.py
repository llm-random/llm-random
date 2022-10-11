import unittest

import numpy as np
import torch


class GeneralTestCase(unittest.TestCase):
    def assertShape(self, tensor, shape):
        self.assertEqual(tuple(tensor.shape), tuple(shape))

    def assertTensorEqual(self, tensor1, tensor2):
        self.assertShape(tensor1, tensor2.shape)
        list1 = list(torch.flatten(tensor1).detach().numpy())
        list2 = list(torch.flatten(tensor2).detach().numpy())
        self.assertListEqual(list1, list2)

    def assertTensorAlmostEqual(self, tensor1, tensor2):
        self.assertShape(tensor1, tensor2.shape)
        list1 = torch.flatten(tensor1).detach().numpy()
        list2 = torch.flatten(tensor2).detach().numpy()
        almostequal = np.isclose(list1, list2,
                                 rtol=1e-5, atol=1e-5)
        listA = list1 * (1-almostequal) + list2 * almostequal
        self.assertListEqual(list(listA), list(list2))

    # def test_assertShape(self):
    #     tensor = torch.rand(2, 3, 4)
    #     self.assertShape(tensor, (2, 3, 4))
    #     self.assertRaises(
    #         AssertionError, self.assertShape, tensor, (2, 3, 5)
    #     )

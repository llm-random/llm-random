import os
import unittest

import numpy as np
import torch


def heavy_test(test):
    def wrapper(*args, **kwargs):
        if os.getenv("SKIP_HEAVY_TESTS"):
            return
        return test(*args, **kwargs)

    return wrapper


def skip_test(reason: str):
    def decorator(test):
        def wrapper(*args, **kwargs):
            print(f"\nSkipping test {test.__qualname__}. Reason: {reason}.")
            return

        return wrapper

    return decorator


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
        almostequal = np.isclose(list1, list2, rtol=1e-5, atol=1e-5)
        listA = list1 * (1 - almostequal) + list2 * almostequal
        self.assertListEqual(list(listA), list(list2))

    # def test_assertShape(self):
    #     tensor = torch.rand(2, 3, 4)
    #     self.assertShape(tensor, (2, 3, 4))
    #     self.assertRaises(
    #         AssertionError, self.assertShape, tensor, (2, 3, 5)
    #     )

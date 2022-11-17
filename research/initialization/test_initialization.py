import torch

from research.initialization import initialization
from lizrd.support.test_utils import GeneralTestCase


class TestPassThrough(GeneralTestCase):
    def test_basic(self):
        batch, dinp = 4, 32
        val_mult, grad_mult = 2.0, 0.5
        layer = initialization.PassThrough(val_mult, grad_mult)
        input = torch.normal(0.0, 1.0, (batch, dinp), requires_grad=True)
        output = layer(input)
        self.assertShape(output, (batch, dinp))
        self.assertTensorEqual(output, input * val_mult)
        loss = output.sum()
        loss.backward()
        self.assertShape(input.grad, (batch, dinp))
        self.assertTensorEqual(input.grad, torch.ones_like(input) * grad_mult)


def std_from_zero(tensor):
    var = tensor * tensor
    mean_var = var.mean()
    return float(torch.sqrt(mean_var))


class TestFixedLinear(GeneralTestCase):
    def layer_init(self, dinp, dout):
        return initialization.FixedLinear(dinp, dout)

    def test_basic(self):
        batch, dinp, dout = 64, 32, 128
        layer = self.layer_init(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, dinp), requires_grad=True)
        output = layer(input)
        params = [p for p in layer.parameters()]
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].shape, (dout, dinp))
        self.assertAlmostEqual(std_from_zero(params[0]), 1.0, delta=0.02)

        # disabled for compression
        self.assertShape(output, (batch, dout))
        self.assertAlmostEqual(std_from_zero(output), 1.0, delta=0.05)
        loss = output.sum()
        loss.backward()
        # disabled for decompression
        # self.assertShape(input.grad, (batch, dinp))
        # self.assertAlmostEqual(std_from_zero(input.grad)), 1.0, delta=0.02)

        # disabled for batch_size>1
        # self.assertShape(params[0].grad, (dout, dinp))
        # self.assertAlmostEqual(std_from_zero(params[0].grad)), 1.0, delta=0.02)

    def test_compression(self):
        batch, dinp, dout = 256 * 8, 256, 64
        layer = self.layer_init(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, dinp), requires_grad=True)
        output = layer(input)
        params = [p for p in layer.parameters()]
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].shape, (dout, dinp))
        self.assertAlmostEqual(std_from_zero(params[0]), 1.0, delta=0.02)

        loss = output.sum()
        loss.backward()
        # disabled for decompression
        self.assertShape(input.grad, (batch, dinp))
        self.assertAlmostEqual(std_from_zero(input.grad), 1.0, delta=0.1)

    def test_straight(self):
        batch, dinp, dout = 256 * 4, 256, 256
        layer = self.layer_init(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, dinp), requires_grad=True)
        output = layer(input)
        params = [p for p in layer.parameters()]
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].shape, (dout, dinp))
        self.assertAlmostEqual(std_from_zero(params[0]), 1.0, delta=0.02)

        self.assertShape(output, (batch, dout))
        self.assertAlmostEqual(std_from_zero(output), 1.0, delta=0.05)
        loss = output.sum()
        loss.backward()
        # disabled for decompression
        self.assertShape(input.grad, (batch, dinp))
        self.assertAlmostEqual(std_from_zero(input.grad), 1.0, delta=0.1)

    def test_singlebatch(self):
        batch, dinp, dout = 1, 1024 * 2, 128 * 2
        layer = self.layer_init(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, dinp), requires_grad=True)
        output = layer(input)
        params = [p for p in layer.parameters()]
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].shape, (dout, dinp))
        self.assertAlmostEqual(std_from_zero(params[0]), 1.0, delta=0.02)

        loss = output.sum()
        loss.backward()

        # disabled for batch_size>1
        self.assertShape(params[0].grad, (dout, dinp))
        self.assertAlmostEqual(std_from_zero(params[0].grad), 1.0, delta=0.05)


class TestStandardLinear(GeneralTestCase):
    def layer_init(self, dinp, dout):
        return initialization.StandardLinear(dinp, dout)

    def test_basic(self):
        batch, dinp, dout = 64, 32, 128
        layer = self.layer_init(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, dinp), requires_grad=True)
        output = layer(input)
        params = [p for p in layer.parameters()]
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].shape, (dout, dinp))
        self.assertAlmostEqual(std_from_zero(params[0]), dinp**-0.5, delta=0.02)

        # disabled for compression
        self.assertShape(output, (batch, dout))
        self.assertAlmostEqual(std_from_zero(output), 1.0, delta=0.05)
        loss = output.sum()
        loss.backward()
        # disabled for decompression
        # self.assertShape(input.grad, (batch, dinp))
        # self.assertAlmostEqual(std_from_zero(input.grad)), 1.0, delta=0.02)

        # disabled for batch_size>1
        # self.assertShape(params[0].grad, (dout, dinp))
        # self.assertAlmostEqual(std_from_zero(params[0].grad)), 1.0, delta=0.02)

    def test_compression(self):
        batch, dinp, dout = 256, 256, 64
        layer = self.layer_init(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, dinp), requires_grad=True)
        output = layer(input)
        params = [p for p in layer.parameters()]
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].shape, (dout, dinp))
        self.assertAlmostEqual(std_from_zero(params[0]), dinp**-0.5, delta=0.001)

    def test_straight(self):
        batch, dinp, dout = 256 * 4, 256, 256
        layer = self.layer_init(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, dinp), requires_grad=True)
        output = layer(input)
        params = [p for p in layer.parameters()]
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].shape, (dout, dinp))
        self.assertAlmostEqual(std_from_zero(params[0]), dinp**-0.5, delta=0.01)

        self.assertShape(output, (batch, dout))
        self.assertAlmostEqual(std_from_zero(output), 1.0, delta=0.05)
        loss = output.sum()
        loss.backward()
        # for straight - this will work
        self.assertShape(input.grad, (batch, dinp))
        self.assertAlmostEqual(std_from_zero(input.grad), 1.0, delta=0.15)

    def test_singlebatch(self):
        batch, dinp, dout = 1, 1024 * 2, 128 * 2
        layer = self.layer_init(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, dinp), requires_grad=True)
        output = layer(input)
        params = [p for p in layer.parameters()]
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].shape, (dout, dinp))
        self.assertAlmostEqual(std_from_zero(params[0]), dinp**-0.5, delta=0.001)

        loss = output.sum()
        loss.backward()

        # disabled for batch_size>1
        self.assertShape(params[0].grad, (dout, dinp))
        self.assertAlmostEqual(std_from_zero(params[0].grad), 1.0, delta=0.05)


# It doesn't exactly work...
# class TestFixedLinearWithReLU(TestFixedLinear):
#     def layer_init(self, dinp, dout):
#         return initialization.FixedLinear(dinp, dout, relu=True)

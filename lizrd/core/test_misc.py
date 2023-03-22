import torch

from lizrd.core import misc
from lizrd.core.misc import Chungus, Checkpoint
from lizrd.support.test_utils import GeneralTestCase


class TestDense(GeneralTestCase):
    def test_basic(self):
        batch, dinp, dout = 4, 32, 64
        layer = misc.DenseEinMix(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dout))

    def test_more_dims(self):
        batch, seql, dinp, dout = 4, 8, 32, 64
        layer = misc.DenseEinMix(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, seql, dinp))
        output = layer(input)
        self.assertShape(output, (batch, seql, dout))


class TestLinear(GeneralTestCase):
    def test_basic(self):
        batch, dinp, dout = 4, 32, 64
        layer = misc.Linear(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dout))

    def test_more_dims(self):
        batch, seql, dinp, dout = 4, 8, 32, 64
        layer = misc.Linear(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, seql, dinp))
        output = layer(input)
        self.assertShape(output, (batch, seql, dout))


class TestEinMix(GeneralTestCase):
    def test_no_ellipsis(self):
        batch, dinp, dout = 4, 32, 64
        layer = misc.EinMix(
            "b d -> b f", weight_shape="d f", bias_shape="f", d=dinp, f=dout
        )
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dout))

    def test_one_ellipsis(self):
        batch, dinp, dout = 4, 32, 64
        layer = misc.EinMix(
            "... d -> ... f", weight_shape="d f", bias_shape="f", d=dinp, f=dout
        )
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dout))

    def test_two_ellipsis(self):
        batch, seqlen, dinp, dout = 4, 2, 32, 64
        layer = misc.EinMix(
            "... d -> ... f", weight_shape="d f", bias_shape="f", d=dinp, f=dout
        )
        input = torch.normal(0.0, 1.0, (batch, seqlen, dinp))
        output = layer(input)
        self.assertShape(output, (batch, seqlen, dout))

    def test_two_ellipsis(self):
        batch, seqlen, whatever, dinp, dout = 5, 7, 3, 32, 64
        layer = misc.EinMix(
            "... d -> ... f", weight_shape="d f", bias_shape="f", d=dinp, f=dout
        )
        input = torch.normal(0.0, 1.0, (batch, seqlen, whatever, dinp))
        output = layer(input)
        self.assertShape(output, (batch, seqlen, whatever, dout))


class TestChungus(GeneralTestCase):
    def test_outputs_and_grads(self):
        torch.manual_seed(0)
        # create a simple Sequential model
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 20),
            torch.nn.Linear(20, 10),
            torch.nn.Linear(10, 20),
        )

        model_chunged = torch.nn.Sequential(
            torch.nn.Linear(100, 20),
            Chungus(torch.nn.Linear(20, 10), n_chungs=2),
            torch.nn.Linear(10, 20),
        )

        # clone the model weights
        for (_, param), (_, param_chunged) in zip(
            model.named_parameters(), model_chunged.named_parameters()
        ):
            param_chunged.data = param.data.clone()

        x = torch.rand(100, 100)
        outputs = {}
        grads = {}
        for model, name in [[model, "vanilla"], [model_chunged, "chunged"]]:
            outputs[name] = model(x)

            model.zero_grad()
            outputs[name].sum().backward()
            grads[name] = {}
            for param_name, param in model.named_parameters():
                grads[name][param_name] = param.grad.data.clone()

        # compare the output and parameters gradients
        assert torch.isclose(
            outputs["vanilla"], outputs["chunged"]
        ).all(), f" output failed, log of difference is: {torch.log10((outputs['vanilla'] - outputs['chunged']).abs().max())}, max difference is: {((output_original - output_checkpointed).abs().max())}, argmax is {((output_original - output_checkpointed).abs().argmax())}"
        for grad, grad_checkpointed in zip(
            grads["vanilla"].values(), grads["chunged"].values()
        ):
            assert torch.isclose(
                grad, grad_checkpointed
            ).all(), f"parameter {name} failed, log of difference is: {torch.log10((grad - grad_checkpointed).abs().max())}"


class TestCheckpoint(GeneralTestCase):
    def test_checkpoint(self):
        torch.manual_seed(0)
        # create a simple Sequential model
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 20),
            torch.nn.Linear(20, 10),
            torch.nn.Linear(10, 20),
        )

        model_checkpointed = torch.nn.Sequential(
            torch.nn.Linear(100, 20),
            Checkpoint(torch.nn.Linear(20, 10)),
            torch.nn.Linear(10, 20),
        )

        # clone the model weights
        for (name, param), (name_checkpointed, param_checkpointed) in zip(
            model.named_parameters(), model_checkpointed.named_parameters()
        ):
            param_checkpointed.data = param.data.clone()

        x = torch.rand(100, 100)
        outputs = {}
        grads = {}
        for model, name in [[model, "vanilla"], [model_checkpointed, "checkpointed"]]:
            outputs[name] = model(x)

            model.zero_grad()
            outputs[name].sum().backward()
            grads[name] = {}
            for param_name, param in model.named_parameters():
                grads[name][param_name] = param.grad.data.clone()

        # compare the output and parameters gradients
        assert torch.isclose(
            outputs["vanilla"], outputs["checkpointed"]
        ).all(), f" output failed, log of difference is: {torch.log10((outputs['vanilla'] - outputs['checkpointed']).abs().max())}, max difference is: {((output_original - output_checkpointed).abs().max())}, argmax is {((output_original - output_checkpointed).abs().argmax())}"
        for grad, grad_checkpointed in zip(
            grads["vanilla"].values(), grads["checkpointed"].values()
        ):
            assert torch.isclose(
                grad, grad_checkpointed
            ).all(), f"parameter {name} failed, log of difference is: {torch.log10((grad - grad_checkpointed).abs().max())}"

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
        # create the model inputs
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
        y = x.clone()
        ###################### ORIGINAL ###########################
        # get the model output and save it to prevent any modifications
        output_original = model(x)

        model.zero_grad()
        output_original.sum().backward()
        grad_vanilla = {}
        for name, param in model.named_parameters():
            grad_vanilla[name] = param.grad.data.clone()

        ###################### CHUNGUS ############################
        output_chunged = model_chunged(y)
        model_chunged.zero_grad()
        output_chunged.sum().backward()

        grad_chunged = {}
        for name, param in model_chunged.named_parameters():
            grad_chunged[name] = param.grad.data.clone()

        ####################### COMPARE ORIG MODEL AND CHUNGUS #######################
        # compare the output and parameters gradients
        assert torch.isclose(
            output_original, output_chunged
        ).all(), f" output failed, log of difference is: {torch.log10((output_original - output_chunged).abs().max())}, max difference is: {((output_original - output_chunged).abs().max())}, argmax is {((output_original - output_chunged).abs().argmax())}"
        for name in grad_vanilla:
            name_for_chunged = name
            if "1" in name:
                name_for_chunged = name.replace("1", "1.module")

            chung = grad_chunged[name_for_chunged]
            van = grad_vanilla[name]

            assert torch.isclose(
                chung, van
            ).all(), f"parameter {name} failed, log of difference is: {torch.log10((chung - van).abs().max())}"


class TestCheckpoint(GeneralTestCase):
    def test_checkpoint(self):
        # create the model inputs
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
        y = x.clone()
        ###################### ORIGINAL ###########################
        # get the model output and save it to prevent any modifications
        output_original = model(x)

        model.zero_grad()
        output_original.sum().backward()
        grad_vanilla = {}
        for name, param in model.named_parameters():
            grad_vanilla[name] = param.grad.data.clone()

        ###################### CHECKPOINT ############################
        output_checkpointed = model_checkpointed(y)
        model_checkpointed.zero_grad()
        output_checkpointed.sum().backward()

        grad_checkpointed = {}
        for name, param in model_checkpointed.named_parameters():
            grad_checkpointed[name] = param.grad.data.clone()

        ####################### COMPARE ORIG MODEL AND CHECKPOINT #######################
        # compare the output and parameters gradients
        assert torch.isclose(
            output_original, output_checkpointed
        ).all(), f" output failed, log of difference is: {torch.log10((output_original - output_checkpointed).abs().max())}, max difference is: {((output_original - output_checkpointed).abs().max())}, argmax is {((output_original - output_checkpointed).abs().argmax())}"
        for name in grad_vanilla:
            name_for_checkpointed = name
            if "1" in name:
                name_for_checkpointed = name.replace("1", "1.module")

            chung = grad_checkpointed[name_for_checkpointed]
            van = grad_vanilla[name]

            assert torch.isclose(
                chung, van
            ).all(), f"parameter {name} failed, log of difference is: {torch.log10((chung - van).abs().max())}"

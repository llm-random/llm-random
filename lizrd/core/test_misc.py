import copy
import torch

from lizrd.core import llm, misc
from lizrd.core.misc import Chungus, Checkpoint
from lizrd.datasets.wikibookdata import get_processed_dataset
from lizrd.support.test_utils import GeneralTestCase, heavy_test, skip_test
from lizrd.train.train_utils import get_model


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

    def test_two_ellipsis_2(self):
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


class TestChungizedCalculateLoss(GeneralTestCase):
    @heavy_test
    @skip_test("Update this test to use the new datasets")
    def test_outputs_and_grads(self):
        torch.manual_seed(0)

        batch, seql, dm, heads, dff = 3, 32, 32, 4, 64
        vocab_size = 30522
        n_blocks = 2
        device = torch.device("cpu")
        mask_percentage = 0.15
        n_chungs = 3

        dataset = get_processed_dataset(
            max_total_length=seql,
            mask_percent=mask_percentage,
            device=device,
            num_workers=1,
            batch_size=batch,
            seed=0,
            model_type="bert",
            use_dummy_dataset=True,
        )

        model = get_model(
            max_length=seql,
            vocab_size=vocab_size,
            ff_layer_fun=lambda: llm.FeedForward(dm, dff),
            attention_layer_fun=lambda: llm.Attention(dm, heads),
            dm=dm,
            n_blocks=n_blocks,
            device=device,
        )

        with torch.no_grad():
            model_chunged = copy.deepcopy(model)

        batch = dataset.get_batch()

        (loss_no_chung, aux_info_no_chung,) = calculate_bert_loss(
            batch=batch, model=model, mixed_precision=False, vocab_size=vocab_size
        )

        (loss_chung, aux_info_chung,) = chungized_bert_loss(
            batch=batch,
            model=model_chunged,
            mixed_precision=False,
            vocab_size=vocab_size,
            n_chungs=n_chungs,
        )

        loss_no_chung.backward()
        loss_chung.backward()
        assert torch.isclose(loss_no_chung, loss_chung)
        assert aux_info_no_chung["correct_tokens"] == aux_info_chung["correct_tokens"]
        assert (
            aux_info_no_chung["total_masked_tokens"]
            == aux_info_chung["total_masked_tokens"]
        )

        chunged_dict = {n: p for n, p in model_chunged.named_parameters()}

        for param_name, param in model.named_parameters():
            assert torch.isclose(param.grad, chunged_dict[param_name].grad).all()


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


class TestModelParallel(GeneralTestCase):
    def test_get_current_device(self):
        obj = llm.TransformerTower(0, 100, {})  # Replace with the name of your class

        # Test when self.splits is None
        obj.model_fragmentation = None
        obj.device = torch.device("cuda")
        assert obj.get_current_device(1) == (False, torch.device("cuda"))

        # Test when self.device is 'cpu'
        obj.model_fragmentation = [1, 2]
        obj.device = torch.device("cpu")
        assert obj.get_current_device(1) == (False, torch.device("cpu"))

        # Test when split_num is greater than block_num:
        obj.model_fragmentation = [3, 5]
        obj.device = torch.device("cuda")
        assert obj.get_current_device(2) == (False, torch.device("cuda:0"))

        # Test when split_num is equal to block_num:
        obj.model_fragmentation = [3, 5]
        obj.device = torch.device("cuda")
        assert obj.get_current_device(3) == (True, torch.device("cuda:1"))

        obj.model_fragmentation = [3, 5]
        obj.device = torch.device("cuda")
        assert obj.get_current_device(5) == (True, torch.device("cuda:2"))

        obj.model_fragmentation = [3, 5]
        obj.device = torch.device("cuda")
        assert obj.get_current_device(4) == (False, torch.device("cuda:1"))

        obj.model_fragmentation = [2]
        obj.device = torch.device("cuda")
        assert obj.get_current_device(2) == (True, torch.device("cuda:1"))

        # Test when none of the split_num is greater than block_num:
        obj.model_fragmentation = [3, 5]
        obj.device = torch.device("cuda")
        assert obj.get_current_device(7) == (False, torch.device("cuda:2"))

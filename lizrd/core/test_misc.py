import copy
import torch

from lizrd.core import llm
from lizrd.core.misc import (
    DenseEinMix,
    Linear,
    EinMix,
)
from lizrd.core.misc import Chungus
from lizrd.support.misc import get_ith_chunk

from research.datasets import get_processed_dataset
from lizrd.support.test_utils import GeneralTestCase, heavy_test
from lizrd.train.train_utils import get_model
from research.conditional.utils.model_utils import (
    calculate_llm_loss_and_gradient,
    chungized_llm_loss_and_gradient,
)


class TestDense(GeneralTestCase):
    def test_basic(self):
        batch, dinp, dout = 4, 32, 64
        layer = DenseEinMix(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dout))

    def test_more_dims(self):
        batch, seql, dinp, dout = 4, 8, 32, 64
        layer = DenseEinMix(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, seql, dinp))
        output = layer(input)
        self.assertShape(output, (batch, seql, dout))


class TestLinear(GeneralTestCase):
    def test_basic(self):
        batch, dinp, dout = 4, 32, 64
        layer = Linear(dinp, dout, init_type="kaiming_uniform", init_scale=1.0)
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dout))

    def test_more_dims(self):
        batch, seql, dinp, dout = 4, 8, 32, 64
        layer = Linear(
            dinp,
            dout,
            init_type="kaiming_uniform",
            init_scale=1.0,
        )
        input = torch.normal(0.0, 1.0, (batch, seql, dinp))
        output = layer(input)
        self.assertShape(output, (batch, seql, dout))


class TestEinMix(GeneralTestCase):
    def test_no_ellipsis(self):
        batch, dinp, dout = 4, 32, 64
        layer = EinMix("b d -> b f", weight_shape="d f", bias_shape="f", d=dinp, f=dout)
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dout))

    def test_one_ellipsis(self):
        batch, dinp, dout = 4, 32, 64
        layer = EinMix(
            "... d -> ... f", weight_shape="d f", bias_shape="f", d=dinp, f=dout
        )
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dout))

    def test_two_ellipsis(self):
        batch, seqlen, dinp, dout = 4, 2, 32, 64
        layer = EinMix(
            "... d -> ... f", weight_shape="d f", bias_shape="f", d=dinp, f=dout
        )
        input = torch.normal(0.0, 1.0, (batch, seqlen, dinp))
        output = layer(input)
        self.assertShape(output, (batch, seqlen, dout))

    def test_two_ellipsis_2(self):
        batch, seqlen, whatever, dinp, dout = 5, 7, 3, 32, 64
        layer = EinMix(
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
    def test_outputs_and_grads(self):
        seed = 0
        torch.manual_seed(seed)

        batch, seql, dm, heads, dff = 3, 32, 32, 4, 64
        vocab_size = 50257
        n_blocks = 2
        device = torch.device("cpu")
        n_chungs = 3

        dataset = get_processed_dataset(
            batch_size=batch,
            sequence_length=seql,
            device=device,
            num_workers=1,
            seed=seed,
            model_type="gpt",
            dataset_type="c4",
            use_dummy_dataset=True,
            dataset_split="train",
        )

        layers = {
            "feedforward": lambda: llm.FeedForward(
                dm, dff, init_type="kaiming_uniform", init_scale=1.0
            ),
            "attention": lambda: llm.Attention(
                dm,
                heads,
                causal=False,
                init_type="kaiming_uniform",
                init_scale=1.0,
            ),
        }

        model = get_model(
            max_length=seql,
            vocab_size=vocab_size,
            block_modules=layers,
            dm=dm,
            n_blocks=n_blocks,
            device=device,
            init_type="kaiming_uniform",
            init_scale=1.0,
            ddp_enabled=False,
            fsdp_enabled=False,
            fsdp_param_precision=None,
            fsdp_mixed_precision_ignore_classes=None,
            fsdp_offload_params=None,
            fsdp_min_num_params=None,
            fsdp_modules_to_wrap=None,
            activation_checkpointing_modules=None,
            is_logging_process=True,
        )

        with torch.no_grad():
            model_chunged = copy.deepcopy(model)

        batch = dataset.get_batch()

        (
            loss_no_chung,
            aux_info_no_chung,
        ) = calculate_llm_loss_and_gradient(
            batch=batch,
            model=model,
            mixed_precision=False,
            mixed_precision_dtype=torch.float16,
            num_checkpoint_accumulation_steps=1,
        )

        (
            loss_chung,
            aux_info_chung,
        ) = chungized_llm_loss_and_gradient(
            batch=batch,
            model=model_chunged,
            mixed_precision=False,
            n_chungs=n_chungs,
            mixed_precision_dtype=torch.float16,
            num_checkpoint_accumulation_steps=1,
        )
        loss_chung = torch.tensor(loss_chung)
        loss_no_chung = torch.tensor(loss_no_chung)
        assert torch.isclose(loss_no_chung, loss_chung)
        assert aux_info_no_chung["correct_tokens"] == aux_info_chung["correct_tokens"]
        assert (
            aux_info_no_chung["total_masked_tokens"]
            == aux_info_chung["total_masked_tokens"]
        )

        chunged_dict = {n: p for n, p in model_chunged.named_parameters()}

        for param_name, param in model.named_parameters():
            assert torch.isclose(param.grad, chunged_dict[param_name].grad).all()

    @heavy_test
    def test_outputs_and_grads_more_accumulation_steps(self):
        seed = 0
        torch.manual_seed(seed)

        batch, seql, dm, heads, dff = 6, 32, 32, 4, 64
        gradient_accumulation_steps = 2
        vocab_size = 50257
        n_blocks = 2
        device = torch.device("cpu")
        n_chungs = 3

        dataset = get_processed_dataset(
            batch_size=batch,
            sequence_length=seql,
            device=device,
            num_workers=1,
            seed=seed,
            model_type="gpt",
            dataset_type="c4",
            use_dummy_dataset=True,
            dataset_split="train",
        )

        layers = {
            "feedforward": lambda: llm.FeedForward(
                dm, dff, init_type="kaiming_uniform", init_scale=1.0
            ),
            "attention": lambda: llm.Attention(
                dm,
                heads,
                causal=False,
                init_type="kaiming_uniform",
                init_scale=1.0,
            ),
        }

        model = get_model(
            max_length=seql,
            vocab_size=vocab_size,
            block_modules=layers,
            dm=dm,
            n_blocks=n_blocks,
            device=device,
            init_type="kaiming_uniform",
            init_scale=1.0,
            ddp_enabled=False,
            fsdp_enabled=False,
            fsdp_param_precision=None,
            fsdp_mixed_precision_ignore_classes=None,
            fsdp_offload_params=None,
            fsdp_min_num_params=None,
            fsdp_modules_to_wrap=None,
            activation_checkpointing_modules=None,
            is_logging_process=True,
        )

        with torch.no_grad():
            model_accumulation = copy.deepcopy(model)

        with torch.no_grad():
            model_chunged_accumulation = copy.deepcopy(model)

        batch = dataset.get_batch()

        (
            normal_loss,
            aux_info_normal,
        ) = calculate_llm_loss_and_gradient(
            batch=batch,
            model=model,
            mixed_precision=False,
            mixed_precision_dtype=torch.float16,
            num_checkpoint_accumulation_steps=1,
        )
        loss_accumulation = 0
        loss_chunged_accumulation = 0
        aux_info_accumulation = {"correct_tokens": 0, "total_masked_tokens": 0}
        aux_info_chunged_accumulation = {"correct_tokens": 0, "total_masked_tokens": 0}
        for i in range(gradient_accumulation_steps):
            batch_copy = copy.deepcopy(batch)
            for _, tensor in batch_copy:
                tensor.data = get_ith_chunk(tensor.data, gradient_accumulation_steps, i)
            (
                loss_accumulation_partial,
                aux_info_partial,
            ) = calculate_llm_loss_and_gradient(
                batch=batch_copy,
                model=model_accumulation,
                mixed_precision=False,
                mixed_precision_dtype=torch.float16,
                num_checkpoint_accumulation_steps=gradient_accumulation_steps,
            )
            loss_accumulation += loss_accumulation_partial
            for k in aux_info_accumulation:
                aux_info_accumulation[k] += aux_info_partial[k]

            (
                loss_accumulation_chunged_partial,
                aux_info_chunged_partial,
            ) = chungized_llm_loss_and_gradient(
                batch=batch_copy,
                model=model_chunged_accumulation,
                mixed_precision=False,
                n_chungs=n_chungs,
                mixed_precision_dtype=torch.float16,
                num_checkpoint_accumulation_steps=gradient_accumulation_steps,
            )
            loss_chunged_accumulation += loss_accumulation_chunged_partial
            for k in aux_info_chunged_accumulation:
                aux_info_chunged_accumulation[k] += aux_info_chunged_partial[k]
        normal_loss = torch.tensor(normal_loss)
        loss_accumulation = torch.tensor(loss_accumulation)
        loss_chunged_accumulation = torch.tensor(loss_chunged_accumulation)
        assert torch.isclose(normal_loss, loss_accumulation)
        assert torch.isclose(normal_loss, loss_chunged_accumulation)
        assert (
            aux_info_normal["correct_tokens"] == aux_info_accumulation["correct_tokens"]
        )
        assert (
            aux_info_normal["correct_tokens"]
            == aux_info_chunged_accumulation["correct_tokens"]
        )
        assert (
            aux_info_normal["total_masked_tokens"]
            == aux_info_accumulation["total_masked_tokens"]
        )
        assert (
            aux_info_normal["total_masked_tokens"]
            == aux_info_chunged_accumulation["total_masked_tokens"]
        )

        accumulation_dict = {n: p for n, p in model_accumulation.named_parameters()}
        chunged_accumulation_dict = {
            n: p for n, p in model_chunged_accumulation.named_parameters()
        }
        for param_name, param in model.named_parameters():
            assert torch.isclose(
                param.grad, chunged_accumulation_dict[param_name].grad
            ).all()
            assert torch.isclose(param.grad, accumulation_dict[param_name].grad).all()


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

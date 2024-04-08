from argparse import Namespace
import unittest

from core.runner import runner
from core.training import StepMetric, TrainingMetricHolder


class TestBasicTraining(unittest.TestCase):
    def test_bacic_training(self):
        args = Namespace(
            all_config_paths="configs/test.yaml",
            attention_mode="vanilla",
            batch_size=10,
            data_seed=1,
            dataset_type="c4",
            dff=32,
            dhead=2,
            dmodel=8,
            embedding_mode="vanilla",
            ff_mode="vanilla",
            final_lr_fraction=0.1,
            final_lr_step=250000,
            flash_attention_enabled=False,
            git_branch="",
            init_scale=1.0,
            init_type="kaiming_uniform",
            learning_rate=0.005,
            lr_scheduler_type="cosine",
            lr_scheduler_warmup_steps=2500,
            model_type="gpt",
            n_att_heads=4,
            n_blocks=2,
            n_gpus=1,
            n_steps=10,
            norm_class="layer_norm",
            num_workers=1,
            optimizer_adam_beta1=0.9,
            optimizer_adam_beta2=0.999,
            optimizer_weight_decay=0.0,
            path_to_entry_config="configs/test.yaml",
            residual_mode="pre_norm",
            seq_length=16,
            tags=[],
            torch_seed=42,
            train_dataset_path=None,
            use_dummy_dataset=True,
            gradient_accumulation_steps=1,
            fsdp_enabled=False
        )
        correct_holder = TrainingMetricHolder()
        metrics = {
            0: StepMetric(12.98619270324707, 672.837646484375),
            1: StepMetric(12.598114013671875, 672.8372802734375),
            2: StepMetric(12.525370597839355, 672.8366088867188),
            3: StepMetric(12.293147087097168, 672.8358154296875),
            4: StepMetric(12.477090835571289, 672.8347778320312),
            5: StepMetric(12.857339859008789, 672.8336181640625),
            6: StepMetric(12.529535293579102, 672.8322143554688),
            7: StepMetric(12.566011428833008, 672.8304443359375),
            8: StepMetric(12.529172897338867, 672.82861328125),
            9: StepMetric(12.516630172729492, 672.8267211914062),
        }
        correct_holder.set_metrics(metrics)
        result = runner(rank=None, args=args, device="cpu")
        self.assertEqual(result, correct_holder)

    def test_gradient_accumulation(self):
        args = Namespace(
            all_config_paths="configs/test.yaml",
            attention_mode="vanilla",
            batch_size=10,
            data_seed=1,
            dataset_type="c4",
            dff=32,
            dhead=2,
            dmodel=8,
            embedding_mode="vanilla",
            ff_mode="vanilla",
            final_lr_fraction=0.1,
            final_lr_step=250000,
            flash_attention_enabled=False,
            git_branch="",
            init_scale=1.0,
            init_type="kaiming_uniform",
            learning_rate=0.005,
            lr_scheduler_type="cosine",
            lr_scheduler_warmup_steps=2500,
            model_type="gpt",
            n_att_heads=4,
            n_blocks=2,
            n_gpus=1,
            n_steps=10,
            norm_class="layer_norm",
            num_workers=1,
            optimizer_adam_beta1=0.9,
            optimizer_adam_beta2=0.999,
            optimizer_weight_decay=0.0,
            path_to_entry_config="configs/test.yaml",
            residual_mode="pre_norm",
            seq_length=16,
            tags=[],
            torch_seed=42,
            train_dataset_path=None,
            use_dummy_dataset=True,
            gradient_accumulation_steps=2,
            fsdp_enabled=False
        )
        correct_holder = TrainingMetricHolder()
        metrics = {
            0: StepMetric(12.986193180084229, 672.837646484375),
            1: StepMetric(12.598114013671875, 672.8372802734375),
            2: StepMetric(12.52537202835083, 672.8366088867188),
            3: StepMetric(12.293147087097168, 672.8358154296875),
            4: StepMetric(12.477090835571289, 672.8347778320312),
            5: StepMetric(12.857339859008789, 672.8336181640625),
            6: StepMetric(12.529535293579102, 672.8322143554688),
            7: StepMetric(12.566009998321533, 672.8304443359375),
            8: StepMetric(12.529172897338867, 672.82861328125),
            9: StepMetric(12.516631126403809, 672.8267211914062)
        }
        correct_holder.set_metrics(metrics)
        result = runner(rank=None, args=args, device="cpu")
        self.assertEqual(result, correct_holder)

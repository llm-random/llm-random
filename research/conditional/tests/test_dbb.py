import torch
from research.conditional.moe_layers.dbb import DBBFF

from lizrd.support.test_utils import GeneralTestCase


class TestDBB(GeneralTestCase):
    def test_eval_equivalence(self):
        """
        Test that DBBFF works the same in train and eval mode.
        """

        batch = 2
        seq_len = 3
        dm = 5
        n_blocks = 7
        dff = 11

        dbb = DBBFF(
            dmodel=dm,
            dff=dff,
            n_blocks=n_blocks,
            init_type="kaiming_uniform",
            init_scale=1.0,
        )

        x = torch.rand(batch, seq_len, dm)
        dbb.train()
        out_train = dbb(x)
        dbb.eval()
        out_eval = dbb(x)
        assert torch.allclose(out_train, out_eval)

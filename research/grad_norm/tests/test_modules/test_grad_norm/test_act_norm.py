import pytest
import torch

from research.grad_norm.modules.grad_norm.act_norm import (
    GradientActivationNormLayer,
    activation_norm_grad,
)


@pytest.fixture
def gn_layer(request) -> GradientActivationNormLayer:
    return GradientActivationNormLayer(norm_dims=request.param)


@pytest.mark.parametrize("gn_layer", [(0, 1, 2), (1, 2), (2,)], indirect=True)
class TestGradNormLayerCommonProperties:
    def test_forward_id(self, gn_layer: GradientActivationNormLayer):

        x = torch.randn(3, 4, 4)
        y = gn_layer(x)

        assert x.shape == y.shape
        assert torch.equal(x, y)

    def test_grad_logging(self, gn_layer: GradientActivationNormLayer):
        gn_layer.update_cache_for_logging = gn_layer.logging_cache.__setitem__

        x = torch.randn(3, 4, 4, requires_grad=True)
        grad = torch.rand_like(x)
        y = gn_layer(x)
        y.backward(grad)

        logs = gn_layer.log_heavy()
        assert torch.equal(logs["activation_norms/mean"], torch.mean(torch.norm(x, dim=-1)))
        assert torch.equal(logs["activation_norms/std"], torch.std(torch.norm(x, dim=-1)))
        assert torch.equal(logs["raw_grad_norms/mean"], torch.mean(torch.norm(grad, dim=-1)))
        assert torch.equal(logs["raw_grad_norms/std"], torch.std(torch.norm(grad, dim=-1)))

    def test_norm_grad_logging(self, gn_layer: GradientActivationNormLayer):
        gn_layer.update_cache_for_logging = gn_layer.logging_cache.__setitem__

        x = torch.randn(3, 4, 4, requires_grad=True)
        grad = torch.rand_like(x)
        y = gn_layer(x)
        y.backward(grad)

        logs = gn_layer.log_heavy()
        norm_grad = torch.norm(activation_norm_grad(grad, x, norm_dims=gn_layer.norm_dims, eps=gn_layer.eps), dim=-1)
        assert torch.equal(logs["norm_grad_norms/mean"], torch.mean(norm_grad))
        assert torch.equal(logs["norm_grad_norms/std"], torch.std(norm_grad))

    def test_backward_norm(self, gn_layer: GradientActivationNormLayer):
        x = torch.randn(3, 4, 5, requires_grad=True)
        grad = torch.randn(3, 4, 5)
        y = gn_layer(x)
        y.backward(grad)

        expected_grad = activation_norm_grad(grad, x, norm_dims=gn_layer.norm_dims, eps=gn_layer.eps)

        assert torch.allclose(x.grad, expected_grad)

import torch

from research.grad_norm.modules.grad_capture import GradCaptureLayer


def test_grad_capture_layer_forward_id():
    layer = GradCaptureLayer()

    x = torch.randn(3, 4, 4)
    y = layer(x)

    assert x.shape == y.shape
    assert torch.equal(x, y)


def test_grad_capture_layer_backward_id():
    layer = GradCaptureLayer()

    x = torch.randn(3, 4, requires_grad=True)
    grad = torch.randn(3, 4)
    y = layer(x)
    y.backward(grad)

    assert torch.allclose(x.grad, grad)


def test_grad_capture_logs():
    layer = GradCaptureLayer()
    layer.update_cache_for_logging = layer.logging_cache.__setitem__

    x = torch.randn(3, 4, requires_grad=True)
    grad = torch.randn(3, 4)
    y = layer(x)
    y.backward(grad)

    logs = layer.log_heavy()
    assert logs["activation_norms/mean"] == torch.mean(torch.norm(x, dim=-1))
    assert logs["activation_norms/std"] == torch.std(torch.norm(x, dim=-1))
    assert logs["grad_norms/mean"] == torch.mean(torch.norm(grad, dim=-1))
    assert logs["grad_norms/std"] == torch.std(torch.norm(grad, dim=-1))
    assert logs["activation_norms/mean"] == torch.mean(torch.norm(x, dim=-1))
    assert logs["activation_norms/std"] == torch.std(torch.norm(x, dim=-1))

from numpy import False_
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from einops import rearrange, repeat

from causal_conv1d import causal_conv1d_fn
import causal_conv1d_cuda
import selective_scan_cuda


class MambaNoOutProjInnerFn(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        xz,
        conv1d_weight,
        conv1d_bias,
        x_proj_weight,
        delta_proj_weight,
        A,
        B=None,
        C=None,
        D=None,
        delta_bias=None,
        B_proj_bias=None,
        C_proj_bias=None,
        delta_softplus=True,
        checkpoint_lvl=1,
    ):
        """
        xz: (batch, dim, seqlen)
        """
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(
                dtype=torch.get_autocast_gpu_dtype()
            )
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
            x, conv1d_weight, conv1d_bias, None, True
        )
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(
            rearrange(conv1d_out, "b d l -> (b l) d"), x_proj_weight
        )  # (bl d)
        delta = rearrange(
            delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L
        )
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank : delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(
                    B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2
                ).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(
                    C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2
                ).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.checkpoint_lvl = checkpoint_lvl
        if (
            checkpoint_lvl >= 1
        ):  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(
            xz,
            conv1d_weight,
            conv1d_bias,
            x_dbl,
            x_proj_weight,
            delta_proj_weight,
            conv1d_out,
            delta,
            A,
            B,
            C,
            D,
            delta_bias,
            scan_intermediates,
            out,
        )
        return out_z  # TODO: might have to rearrange this

    @staticmethod
    @custom_bwd
    def backward(ctx, dout_y):
        # dout: (batch, seqlen, dim)
        (
            xz,
            conv1d_weight,
            conv1d_bias,
            x_dbl,
            x_proj_weight,
            delta_proj_weight,
            conv1d_out,
            delta,
            A,
            B,
            C,
            D,
            delta_bias,
            scan_intermediates,
            out,
        ) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout_y.stride(-1) != 1:
            dout_y = dout_y.contiguous()  # might have to rearrange dout_y
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
                x, conv1d_weight, conv1d_bias, None, True
            )
            delta = rearrange(
                delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L
            )
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        (
            dconv1d_out,
            ddelta,
            dA,
            dB,
            dC,
            dD,
            ddelta_bias,
            dz,
            _,
        ) = selective_scan_cuda.bwd(
            conv1d_out,
            delta,
            A,
            B,
            C,
            D,
            z,
            delta_bias,
            dout_y,
            scan_intermediates,
            out,
            dz,
            ctx.delta_softplus,
            False,  # option to recompute out_z
        )
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(
                    dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2
                ).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank : delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(
                    dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2
                ).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum(
            "Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d")
        )
        dconv1d_out = torch.addmm(
            dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out
        )
        dconv1d_out = rearrange(
            dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1]
        )
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, dx, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (
            dxz,
            dconv1d_weight,
            dconv1d_bias,
            dx_proj_weight,
            ddelta_proj_weight,
            dA,
            dB,
            dC,
            dD,
            ddelta_bias if delta_bias is not None else None,
            dB_proj_bias,
            dC_proj_bias,
            None,
        )


def mamba_no_out_proj_inner_fn(
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    A,
    B=None,
    C=None,
    D=None,
    delta_bias=None,
    B_proj_bias=None,
    C_proj_bias=None,
    delta_softplus=True,
):
    return MambaNoOutProjInnerFn.apply(
        xz,
        conv1d_weight,
        conv1d_bias,
        x_proj_weight,
        delta_proj_weight,
        A,
        B,
        C,
        D,
        delta_bias,
        B_proj_bias,
        C_proj_bias,
        delta_softplus,
    )

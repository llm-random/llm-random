@ash.check("... d -> ... d")
class BatchSplitFF(nn.Module):
    def __init__(
        self,
        register_list,
        dm,
        dff,
        expertsets,
        nexperts,
        expertsize,
        controller_loss_weight=1.0,
    ):
        super(BatchSplitFF, self).__init__()
        # register_list will be used, together with some get_loss function, to compute loss
        # this will require gradients to be already in place!
        register_list.append(self)

        assert dff == expertsets * nexperts * expertsize
        self.dm = dm
        self.dff = dff
        self.expertsets = expertsets
        self.nexperts = nexperts
        self.expertsize = expertsize

        # assert expertsets == nexperts  # TODO: remove, it shouldn't be necessary

        self.controller = nn.Parameter(
            torch.Tensor(
                dm,
                nexperts,
                expertsets,
            )
        )
        self.cp = "d e s"
        self.gp = "... t d"
        self.cout = "... t e s"
        self.inner = "... e s f"

        self.new_parameter = nn.Parameter(torch.Tensor(dm))

        self.bias = nn.Parameter(torch.Tensor(nexperts, expertsets, expertsize))

        self.f1p = "d e s f"
        self.f1 = nn.Parameter(torch.Tensor(dm, nexperts, expertsets, expertsize))

        self.f2p = "e s f d"
        self.f2 = nn.Parameter(torch.Tensor(nexperts, expertsets, expertsize, dm))

        self.inner2 = "... e s d"

        self.ogp = self.gp.replace("...", "... b")
        self.controller_loss_weight = controller_loss_weight

        self.register_full_backward_hook(BatchSplitFF.backward_hook_batch_split_ff)
        self.last_x = None

    def backward_hook_batch_split_ff(self, grad_input, grad_output):
        # for now we completely ignore which experts were activated etc.
        x = self.last_x.detach()
        grad_output = grad_output[0].detach()
        del grad_input
        with torch.enable_grad():
            wtf = self.new_parameter
            x = x + wtf
            print(f"wtf.requires_grad = {wtf.requires_grad}")
            print(f"x.requires_grad = {x.requires_grad}")
            something = einsum("... d, ... d -> ...", x, grad_output)
            something = torch.mean(something)
            # something.requires_grad = True

            print(f"something: {torch.mean(something)}")
            print(f"grad_output[0]: {torch.mean(grad_output[0])}")
            print(f"x: {torch.mean(x)}")
            print(f"shape of something: {something.shape}")
            print(f"shape of grad_output[0]: {grad_output[0].shape}")
            print(f"shape of x: {x.shape}")

            print(f"self.new_parameter.grad: {self.new_parameter.grad}")
            something.backward()
            print(f"self.new_parameter.grad: {self.new_parameter.grad}")
            exit(0)

        # grouped: ... b t d

        # print('backward hook batch split ff')
        # print(grad_input)
        # print(grad_output)
        # print(self.f1.grad)
        # print(self.f2.grad)
        # print(self.bias.grad)
        # print(self.controller.grad)
        # print('\n')
        pass

    def forward(self, x):
        self.last_x = x.detach()
        # TODO: I want to, if model is in .train(), then also run all things
        # that we need for
        with Timer("batchedFF", disable_inner=False):
            # BATCH, embedding
            ash.assert_shape("... B d", x, d=self.dm)
            # batch, set, embedding <-- this is just reshape
            grouped = einops.rearrange(x, f"... (b t) d -> {self.ogp}", t=self.nexperts)

            ## CONTROLLER:
            # batch, set1, embedding <-- this is starting point
            cont_logits = einsum(
                f"{self.gp}, {self.cp} -> {self.cout}", grouped, self.controller
            )
            # biases in the controller are not needed, because they are added to
            # every token in a given expert, and expert chooses the token with max value

            # batch, set1, set2(experts), expertsets  <--- this comes from linear
            # batch, set1, set2(experts), expertsets <--- sample on 1st dimension (set1)
            # In lieu of adding noise, we can prioritize earlier tokens. This breaks symmetry.

            cont_logits += torch.reshape(
                torch.linspace(
                    start=0, end=1e-6, steps=self.nexperts, device=x.device
                ),  # to break symmetry
                (-1, 1, 1),
            )
            # cont_permutation = cont_logits
            cont_permutation = torch.eq(
                cont_logits, torch.max(cont_logits, dim=-3, keepdim=True)[0]
            )
            cont_permutation = cont_permutation * 1.0  # convert to float tensor

            inner = einsum(
                f"{self.gp}, {self.f1p}, {self.cout} -> {self.inner}",
                grouped,
                self.f1,
                cont_permutation,
                use_opt_einsum=True,
            )

            inner = inner + self.bias

            inner = torch.relu_(inner)

            intermediate = einsum(
                f"{self.inner},{self.f2p} -> {self.inner2}", inner, self.f2
            )
            result_unpermuted = einsum(
                f"{self.inner2}, {self.cout} -> {self.gp}",
                intermediate,
                cont_permutation,
            )

            # final reshape
            # BATCH, embedding
            result_final = einops.rearrange(
                result_unpermuted, f"{self.ogp} -> ... (b t) d"
            )

            return result_final

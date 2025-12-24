import torch
import torch.nn as nn
import numpy as np
from models.embedder import get_embedder



##########Define Bone_reconstruction_UDF udf Network#################
class UltraBoneUDF(nn.Module):
    def __init__(self,
                 dims=[3,64,128,128,256,256,256,1],
                 skip_in=(4,),
                 bias=0.5,
                 scale=1,
                 multires=0,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 ):
        super(UltraBoneUDF, self).__init__()


        self.embed_fn_fine = None
        if multires > 0:
            print('PE')
            embed_fn, input_ch = get_embedder(multires, input_dims=dims[0])
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in


        for l in range(0, self.num_layers - 1):

            in_dim = dims[l]
            out_dim = dims[l + 1]
            lin = nn.Linear(in_dim, out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)

        self.activation = nn.ReLU()






    def forward(self, query_pcd):
        query_pcd = query_pcd
        if self.embed_fn_fine is not None:
            query_pcd = self.embed_fn_fine(query_pcd)

        x = query_pcd

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)

        return torch.abs(x)


    def udf(self, query_pcd):
        return self.forward(query_pcd)

    def udf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, query_pcd):
        query_pcd.requires_grad_(True)
        y = self.udf(query_pcd)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=query_pcd,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)





import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Ring-GNN equi 2 to 2 layer file
    On the equivalence between graph isomorphism testing and function approximation with GNNs (Chen et al, 2019)
    https://arxiv.org/pdf/1905.12560v1.pdf

    CODE ADPATED FROM https://github.com/leichen2018/Ring-GNN/
"""
    
class RingGNNEquivLayer(nn.Module):
    def __init__(self, device, input_dim, output_dim, layer_norm, residual, dropout,
                 normalization='inf', normalization_val=1.0, radius=2, k2_init = 0.1):
        super().__init__()
        self.device = device
        basis_dimension = 15
        self.radius = radius
        self.layer_norm = layer_norm
        self.residual = residual
        self.dropout = dropout
        
        coeffs_values = lambda i, j, k: torch.randn([i, j, k]) * torch.sqrt(2. / (i + j).float())
        self.diag_bias_list = nn.ParameterList([])

        for i in range(radius):
            for j in range(i+1):
                self.diag_bias_list.append(nn.Parameter(torch.zeros(1, output_dim, 1, 1)))

        self.all_bias = nn.Parameter(torch.zeros(1, output_dim, 1, 1))
        self.coeffs_list = nn.ParameterList([])

        for i in range(radius):
            for j in range(i+1):
                self.coeffs_list.append(nn.Parameter(coeffs_values(input_dim, output_dim, basis_dimension)))

        self.switch = nn.ParameterList([nn.Parameter(torch.FloatTensor([1])), nn.Parameter(torch.FloatTensor([k2_init]))])
        self.output_dim = output_dim

        self.normalization = normalization
        self.normalization_val = normalization_val
        
        if self.layer_norm:
            self.ln_x = LayerNorm(output_dim.item())
            
        if self.residual:
            self.res_x = nn.Linear(input_dim.item(), output_dim.item())

    def forward(self, inputs):
        m = inputs.size()[3]
        
        ops_out = ops_2_to_2(inputs, m, normalization=self.normalization)
        ops_out = torch.stack(ops_out, dim = 2)


        output_list = []

        for i in range(self.radius):
            for j in range(i+1):
                output_i = torch.einsum('dsb,ndbij->nsij', self.coeffs_list[i*(i+1)//2 + j], ops_out)

                mat_diag_bias = torch.eye(inputs.size()[3]).unsqueeze(0).unsqueeze(0).to(self.device) * self.diag_bias_list[i*(i+1)//2 + j]
                # mat_diag_bias = torch.eye(inputs.size()[3]).to('cuda:0').unsqueeze(0).unsqueeze(0) * self.diag_bias_list[i*(i+1)//2 + j]
                if j == 0:
                    output = output_i + mat_diag_bias
                else:
                    output = torch.einsum('abcd,abde->abce', output_i, output)
            output_list.append(output)

        output = 0
        for i in range(self.radius):
            output += output_list[i] * self.switch[i]

        output = output + self.all_bias
        
        if self.layer_norm:
            # Now, changing shapes from [1xdxnxn] to [nxnxd] for BN
            output = output.permute(3,2,1,0).squeeze()
            
            # output = self.bn_x(output.reshape(m*m, self.output_dim.item())) # batch normalization
            output = self.ln_x(output)   # layer normalization
            
            # Returning output back to original shape
            output = output.reshape(m, m, self.output_dim.item())
            output = output.permute(2,1,0).unsqueeze(0)
        
        output = F.relu(output) # non-linear activation
        
        if self.residual:
            # Now, changing shapes from [1xdxnxn] to [nxnxd] for Linear() layer
            inputs, output = inputs.permute(3,2,1,0).squeeze(), output.permute(3,2,1,0).squeeze()
            
            residual_ = self.res_x(inputs)
            output = residual_ + output # residual connection
            
            # Returning output back to original shape
            output = output.permute(2,1,0).unsqueeze(0)
            
        output = F.dropout(output, self.dropout, training=self.training)
        
        return output


def ops_2_to_2(inputs, dim, normalization='inf', normalization_val=1.0): # N x D x m x m
    # input: N x D x m x m
    diag_part = torch.diagonal(inputs, dim1 = 2, dim2 = 3) # N x D x m
    sum_diag_part = torch.sum(diag_part, dim=2, keepdim = True) # N x D x 1
    sum_of_rows = torch.sum(inputs, dim=3) # N x D x m
    sum_of_cols = torch.sum(inputs, dim=2) # N x D x m
    sum_all = torch.sum(sum_of_rows, dim=2) # N x D

    # op1 - (1234) - extract diag
    op1 = torch.diag_embed(diag_part) # N x D x m x m
    
    # op2 - (1234) + (12)(34) - place sum of diag on diag
    op2 = torch.diag_embed(sum_diag_part.repeat(1, 1, dim))
    
    # op3 - (1234) + (123)(4) - place sum of row i on diag ii
    op3 = torch.diag_embed(sum_of_rows)

    # op4 - (1234) + (124)(3) - place sum of col i on diag ii
    op4 = torch.diag_embed(sum_of_cols)

    # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
    op5 = torch.diag_embed(sum_all.unsqueeze(2).repeat(1, 1, dim))
    
    # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
    op6 = sum_of_cols.unsqueeze(3).repeat(1, 1, 1, dim)
    
    # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
    op7 = sum_of_rows.unsqueeze(3).repeat(1, 1, 1, dim)

    # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
    op8 = sum_of_cols.unsqueeze(2).repeat(1, 1, dim, 1)

    # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
    op9 = sum_of_rows.unsqueeze(2).repeat(1, 1, dim, 1)

    # op10 - (1234) + (14)(23) - identity
    op10 = inputs

    # op11 - (1234) + (13)(24) - transpose
    op11 = torch.transpose(inputs, -2, -1)

    # op12 - (1234) + (234)(1) - place ii element in row i
    op12 = diag_part.unsqueeze(3).repeat(1, 1, 1, dim)

    # op13 - (1234) + (134)(2) - place ii element in col i
    op13 = diag_part.unsqueeze(2).repeat(1, 1, dim, 1)

    # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
    op14 = sum_diag_part.unsqueeze(3).repeat(1, 1, dim, dim)

    # op15 - sum of all ops - place sum of all entries in all entries
    op15 = sum_all.unsqueeze(2).unsqueeze(3).repeat(1, 1, dim, dim)

    #A_2 = torch.einsum('abcd,abde->abce', inputs, inputs)
    #A_4 = torch.einsum('abcd,abde->abce', A_2, A_2)
    #op16 = torch.where(A_4>1, torch.ones(A_4.size()), A_4)

    if normalization is not None:
        float_dim = float(dim)
        if normalization is 'inf':
            op2 = torch.div(op2, float_dim)
            op3 = torch.div(op3, float_dim)
            op4 = torch.div(op4, float_dim)
            op5 = torch.div(op5, float_dim**2)
            op6 = torch.div(op6, float_dim)
            op7 = torch.div(op7, float_dim)
            op8 = torch.div(op8, float_dim)
            op9 = torch.div(op9, float_dim)
            op14 = torch.div(op14, float_dim)
            op15 = torch.div(op15, float_dim**2)
    
    #return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16]
    '''
    l = [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]
    for i, ls in enumerate(l):
        print(i+1)
        print(torch.sum(ls))
    print("$%^&*(*&^%$#$%^&*(*&^%$%^&*(*&^%$%^&*(")
    '''
    return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]    


class LayerNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Parameter(torch.ones(d).unsqueeze(0).unsqueeze(0)) # shape is 1 x 1 x d
        self.b = nn.Parameter(torch.zeros(d).unsqueeze(0).unsqueeze(0)) # shape is 1 x 1 x d
        
    def forward(self, x):
        # x tensor of the shape n x n x d
        mean = x.mean(dim=(0,1), keepdim=True)
        var = x.var(dim=(0,1), keepdim=True, unbiased=False)
        x = self.a * (x - mean) / torch.sqrt(var + 1e-6) + self.b # shape is n x n x d
        return x

        
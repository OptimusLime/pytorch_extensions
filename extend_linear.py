import os
import torch
import torch.nn.init as init
from ipdb import set_trace as bb
import numpy as np
import math


def xavier_init(weight, bias=None):
    bound = 1 / math.sqrt(weight.size(1))
    init.uniform_(weight, -bound, bound)
    if bias is not None:
        init.uniform_(bias, -bound, bound)


def get_expanded_index(sel_ixs, expand_shape, add_count=0):
    base_vals = sel_ixs
    if add_count > 0:
        base_vals = torch.cat([base_vals, base_vals.new_zeros(add_count)])
    
    full_size = base_vals.shape[0]
    # need to be (1s) for uniportant dimensions
    # then -1 for the dimension we want to expand from
    if full_size == 1:
        view_shape = [1 for _ in expand_shape]
    else:
        view_shape = [1 if shape != full_size else -1
                    for shape in expand_shape]

    return base_vals.view(view_shape).expand(expand_shape)


def extend_linear(linear, add_inputs=0, add_outputs=0):
    weight, bias = linear.weight.detach(), linear.bias.detach()
    new_w, new_b = extended_weights(
                        weight, bias, 
                        add_inputs=add_inputs,
                        add_outputs=add_outputs)

    out_count, in_count = new_w.shape[-2:]
    # create our expanded universe
    expanded = torch.nn.Linear(
        in_count, out_count,
        ).type(weight.type())
    # replace our weights and bias
    expanded.weight.data.copy_(new_w.squeeze(0))
    expanded.bias.data.copy_(new_b.squeeze(0))

    return expanded

def extended_weights(weight, bias, 
                  add_inputs=0, 
                  add_outputs=0,
                  init_inputs=xavier_init,
                  init_outputs=xavier_init):
                #   keep_inputs=None,
                #   exclude_inputs=None,
                #   keep_outputs=None,
                #   exclude_outputs=None,


    # expand if necessary
    no_batch = weight.dim() <= 2
    weight = weight.unsqueeze(0) if no_batch else weight
    bias = bias.unsqueeze(0) if no_batch else bias

    # now we have all proper dimensions
    batch_count, out_count, in_count = weight.shape

    expand_shape = (batch_count, 
                    out_count + add_outputs,
                    in_count + add_inputs) 

    batch_ixs = get_expanded_index(
        torch.arange(batch_count), expand_shape)

    out_ixs = get_expanded_index(
        torch.arange(out_count), expand_shape, add_outputs
    )
    
    in_ixs = get_expanded_index(
        torch.arange(in_count), expand_shape, add_inputs
    )

    # select our weights
    new_weight = weight[batch_ixs, out_ixs, in_ixs]
    new_bias = bias[batch_ixs[...,0], out_ixs[...,0]]

    # then we need to init in place the new weights
    if add_outputs > 0:
        init_outputs(new_weight[..., -add_outputs:, :], 
                     bias= new_bias[..., -add_outputs:])

    if add_inputs > 0:
        init_inputs(new_weight[..., :, -add_inputs:])

    # send it back
    return new_weight, new_bias

    # any_remove = False
    # # if we want to remove inputs, put them in keep/exlude
    # if keep_inputs is not None or exclude_inputs is not None:
    #     # we want to remove inputs
    #     any_remove = True
    #     pass
    
    # if keep_outputs is not None or exclude_outputs is not None:
    #     any_remove = True
    #     # we want to remove outputs
    #     pass

    # new_weight = linear.weight
    # new_bias = linear.bias

    # # 
    # if any_remove:
    #     raise NotImplementedError("implement removing features")

    # # okay, now we want to add, this is easy
    # # we append inputs and outputs appropriately
    # # outputs are easiest to initialize
    # if add_outputs > 0:
    #     # create our outputs
    #     outputs = new_weight.new([add_outputs, new_weight.shape[-1]])
    #     bias = new_bias.new([add_outputs])

    #     if init_weights is None:
    #         xavier_init(outputs, bias)
    #     else:
    #         # initialize weights in place
    #         # note that these are not inputs 
    #         # (potentially different init)
    #         init_weights(outputs, bias, is_input=False)

    #     if new_weight.dim() > 2:
    #         # batch dimension in the weights
    #         outputs = outputs.unsqueeze(0)
    #         bias = bias.unsqueeze(0)

    #     # now cat along first dimension
    #     new_weight = torch.cat([new_weight, outputs], -2)
    #     new_bias = torch.cat([new_bias, bias], -1)

    # # now we need to add inputs 
    # if add_inputs > 0:
    #     # we'll need to expand
    #     inputs = new_weight.new([new_weight.shape[-2], add_inputs])

    #     if init_weights is None:
    #         xavier_init(inputs)
    #     else:
    #         # init new inputs in place
    #         # inputs might be init to smaller values to avoid impact
    #         init_weights(inputs, is_input=True)

    #     # init the weights
    #     if new_weight.dim() > 2:
    #         # expand dim
    #         inputs = inputs.unsqueeze(0)

    #     # ready to append column
    #     new_weight = torch.cat([new_weight, inputs], -1)
    
    # # now we're ready to create our weights
    # return new_weight, new_bias


def ri(min_v, max_v):
    return torch.randint(1, 10, [1]).item()

if __name__ == "__main__":

    in_count, out_count = ri(1, 10), ri(1, 10)
    bs = 10
    # create our inputs
    inputs = torch.randn(bs, in_count)

    # create our linear network
    linear_base = torch.nn.Linear(in_count, out_count)

    # get our original
    base_outputs = linear_base(inputs)

    # now let's do some expansion tests
    # v1, add some outputs
    add_out = ri(1,3)

    # create a new linear layer with more outputs
    linear_addout = extend_linear(linear_base, add_outputs=add_out)
    
    # we should have identical behavior
    expand_out = linear_addout(inputs)

    assert (base_outputs - expand_out[...,:-add_out]).sum().item() == 0, \
        "Base output calculations should be identical!"
    
    # 
    bb()

    print("Finished testing linear extension")


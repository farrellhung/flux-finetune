
# import torch
# from torch.functional import *

# from typing import Callable, List, Optional, Tuple, Union

# try:
#     import numpy as np
# except ModuleNotFoundError:
#     np = None

# import torch
# from torch import _VF
# from torch import sym_int as _sym_int
# from torch._C import _infer_size, _add_docstr
# from torch._torch_docs import reproducibility_notes, tf32_notes, sparse_support_notes
# # A workaround to support both TorchScript and MyPy:
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from torch.types import _dtype as DType
# else:
#     # The JIT doesn't understand Union, nor torch.dtype here
#     DType = int

# from torch.overrides import (
#     has_torch_function, has_torch_function_unary, has_torch_function_variadic,
#     handle_torch_function)
# from torch.nn import _reduction as _Reduction
# from torch.nn import grad  # noqa: F401
# from torch.nn.modules import utils
# from torch.nn.modules.utils import _single, _pair, _triple, _list_with_default


# Tensor = torch.Tensor

# def cross_entropy(
#     input: Tensor,
#     target: Tensor,
#     weight: Optional[Tensor] = None,
#     size_average: Optional[bool] = None,
#     ignore_index: int = -100,
#     reduce: Optional[bool] = None,
#     reduction: str = "mean",
#     label_smoothing: float = 0.0,
# ) -> Tensor:
#     r"""Compute the cross entropy loss between input logits and target.

#     See :class:`~torch.nn.CrossEntropyLoss` for details.

#     Args:
#         input (Tensor) : Predicted unnormalized logits;
#             see Shape section below for supported shapes.
#         target (Tensor) : Ground truth class indices or class probabilities;
#             see Shape section below for supported shapes.
#         weight (Tensor, optional): a manual rescaling weight given to each
#             class. If given, has to be a Tensor of size `C`
#         size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
#             the losses are averaged over each loss element in the batch. Note that for
#             some losses, there multiple elements per sample. If the field :attr:`size_average`
#             is set to ``False``, the losses are instead summed for each minibatch. Ignored
#             when reduce is ``False``. Default: ``True``
#         ignore_index (int, optional): Specifies a target value that is ignored
#             and does not contribute to the input gradient. When :attr:`size_average` is
#             ``True``, the loss is averaged over non-ignored targets. Note that
#             :attr:`ignore_index` is only applicable when the target contains class indices.
#             Default: -100
#         reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
#             losses are averaged or summed over observations for each minibatch depending
#             on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
#             batch element instead and ignores :attr:`size_average`. Default: ``True``
#         reduction (str, optional): Specifies the reduction to apply to the output:
#             ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
#             ``'mean'``: the sum of the output will be divided by the number of
#             elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
#             and :attr:`reduce` are in the process of being deprecated, and in the meantime,
#             specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
#         label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount
#             of smoothing when computing the loss, where 0.0 means no smoothing. The targets
#             become a mixture of the original ground truth and a uniform distribution as described in
#             `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.

#     Shape:
#         - Input: Shape :math:`(C)`, :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
#           in the case of `K`-dimensional loss.
#         - Target: If containing class indices, shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with
#           :math:`K \geq 1` in the case of K-dimensional loss where each value should be between :math:`[0, C)`.
#           If containing class probabilities, same shape as the input and each value should be between :math:`[0, 1]`.

#         where:

#         .. math::
#             \begin{aligned}
#                 C ={} & \text{number of classes} \\
#                 N ={} & \text{batch size} \\
#             \end{aligned}

#     Examples::

#         >>> # Example of target with class indices
#         >>> input = torch.randn(3, 5, requires_grad=True)
#         >>> target = torch.randint(5, (3,), dtype=torch.int64)
#         >>> loss = F.cross_entropy(input, target)
#         >>> loss.backward()
#         >>>
#         >>> # Example of target with class probabilities
#         >>> input = torch.randn(3, 5, requires_grad=True)
#         >>> target = torch.randn(3, 5).softmax(dim=1)
#         >>> loss = F.cross_entropy(input, target)
#         >>> loss.backward()
#     """
#     if has_torch_function_variadic(input, target, weight):
#         return handle_torch_function(
#             cross_entropy,
#             (input, target, weight),
#             input,
#             target,
#             weight=weight,
#             size_average=size_average,
#             ignore_index=ignore_index,
#             reduce=reduce,
#             reduction=reduction,
#             label_smoothing=label_smoothing,
#         )
#     if size_average is not None or reduce is not None:
#         reduction = _Reduction.legacy_get_string(size_average, reduce)
#     return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)


import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

def orthogonal_regularization(input, lambda_ortho=8e-8):
    transpose = torch.transpose(input,2,3)
    inner = torch.matmul(transpose, input)
    eye = torch.eye(inner.size(2), device=input.device)
    # eye = torch.diag(torch.flatten(row_norms ** 2))
    
    ortho_loss = torch.norm(inner - eye, p='fro')
    ortho_loss = (ortho_loss**2)*lambda_ortho

    f = open('ortho_loss.txt', 'a')
    f.write("{:.6f}".format(float(ortho_loss))+'\n')
    f.close()

    return ortho_loss

def basis_regularization(input, lambda_basis=4e-7):
    basis_loss = torch.sum((torch.sum(input**2, dim=2) - 1)**2)
    basis_loss = basis_loss*lambda_basis

    f = open('basis_loss.txt', 'a')
    f.write("{:.6f}".format(float(basis_loss))+'\n')
    f.close()

    return basis_loss
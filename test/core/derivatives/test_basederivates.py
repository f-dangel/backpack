"""Test class for module partial derivatives.

- Jacobian-matrix products
- transposed Jacobian-matrix products
"""

import warnings

import pytest
import torch


from backpack import extend

import backpack.core.derivatives.linear
import backpack.core.derivatives.basederivatives
from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.core.derivatives import derivatives_for
from backpack.hessianfree.rop import jacobian_vector_product


from test.core.derivatives.utils import check_sizes, check_values, get_available_devices
from test.core.derivatives.test_setup import ALL_CONFIGURATIONS, CONFIGURATION_IDS

torch.manual_seed(123)
        

class DerivativesImplementation:
    """Base class for autograd and BackPACK implementations.

    self.input_mat: The input matrix required for testing is initialized
    """

    def __init__(self, derivatives_module):
        self.N = derivatives_module.N
        self.input_shape = derivatives_module.input_shape 
        self.module = derivatives_module.module
        self.g_inp, self.g_out = None, None

        #Initialize input_mat
        torch.manual_seed(123)
        self.input_mat = torch.rand(self.N, *self.input_shape) 
        

    def jac_t_mat_prod(self, mat):        
        raise NotImplementedError



class BackpackDerivativesImplementation(DerivativesImplementation):
    """Derivative implementations with BackPACK.
    """
    def __init__(self, derivatives_module):
        super().__init__(derivatives_module)
        self.module = extend(self.module) 

        def get_backpack_class(module):
            '''
            Input:
                module: torch.nn module 
            Returns:
                layer_backpack_class = class instance of torch.nn from 'backpack.core.derivatives`

            '''
            if isinstance(module,type(module)):  
                layer_backpack_class = derivatives_for[type(module)]         
                return layer_backpack_class

            raise RuntimeError("No derivative available for {}".format(module))

        self.output_mat = self.module(self.input_mat)
        
        layer_backpack_class = get_backpack_class(self.module)
        self.backpackClass = layer_backpack_class()


    def jac_t_mat_prod(self, mat):
        '''
        Input:
            mat: Matrix with which the jacobian is multiplied.
                 shape: [V, N, C_in, H_in, W_in]
        Return:
            jmp: Jacobian matrix product obatined from backPACK

        '''
        jmp = self.backpackClass.jac_mat_prod(self.module, self.g_inp, self.g_out,mat)
        return jmp


class AutogradDerivativesImplementation(DerivativesImplementation):
    """Derivative implementations with autograd.

    """
    def __init__(self, derivatives_module):
        super().__init__(derivatives_module)

    def jac_t_mat_prod_vec(self, input_mat, mat):
        '''
        Input:
            input_mat: input matrix.
                       shape: [N, C, H_in, W_in]
                  mat: Matrix with which the jacobian is multiplied.
                       shape: [N, C_in, H_in, W_in]
        Returns:
             jmp_vec: Jacobian matrix product obatined from torch.autograd
        '''
        input_mat.requires_grad = True
        output_mat = self.module(input_mat)
        jmp_vec = jacobian_vector_product(output_mat, self.input_mat, mat)
        return jmp_vec[0] #returns as a tuple?


    def jac_t_mat_prod(self, mat):
        '''
        Input:
            mat: Matrix with which the jacobian is multiplied.
                 shape: [V, N, C_in, H_in, W_in]
        Return:
            jmp: Jacobian matrix product obatined from torch.autograd

        '''
        V = mat.shape[0]
        jmp_list = []

        for v in range(V):
            mat_ = mat[v]
            jmp_mat = self.jac_t_mat_prod_vec(self.input_mat, mat_)
            jmp_list.append(jmp_mat)

        jmp = torch.stack(jmp_list)
        return jmp



@pytest.mark.parametrize(
    "derivatives_module", ALL_CONFIGURATIONS, ids=CONFIGURATION_IDS)

def test_jac_t_mat_prod(derivatives_module, V=10):
    """Test the transposed Jacobian-matrix product by comparing the 
        values and sizes obatined by backpack with torch.autograd 
    
    mat: Matrix with which the jacobian is multiplied.
         shape: [V, N, C_in, H_in, W_in]
                V: vec_cols
                N: batch_size
                C_in: Channels
                H_in, W_in: Height, Width 


    """
    N = derivatives_module.N
    input_shape = derivatives_module.input_shape    

    torch.manual_seed(123)
    mat = torch.rand(V, N, *input_shape) 

    
    backpack_res = BackpackDerivativesImplementation(derivatives_module).jac_t_mat_prod(
        mat
    )

    autograd_res = AutogradDerivativesImplementation(derivatives_module).jac_t_mat_prod(
        mat
    )

    check_sizes(autograd_res, backpack_res)
    check_values(autograd_res, backpack_res)



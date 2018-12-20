"""
Transform layers into parallel series leading to subgrouping of parameters.
This allows to further break the block-diagonal approximation of the Hessian
apart, leading to smaller block structures.
"""

from backpack.utils.utils import einsum
from backpack.utils import conv as convUtils
from backpack.new_extensions.firstorder import FirstOrderExtension


class BatchL2Conv2d(FirstOrderExtension):
    def __init__(self):
        super().__init__(params=["bias", "weight"])


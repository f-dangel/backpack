"""Modification of second-order module effects during Hessian backpropagation.

The residual term is tweaked to give rise to the following curvatures:
- No modification: Exact Hessian
- Neglect module second order information: Generalized Gauss-Newton matrix
- Cast negative residual eigenvalue to their absolute value: PCH-abs
- Set negative residual eigenvalues to zero: PCH-clip
"""


class ResidualModifications:
    @staticmethod
    def nothing(res):
        return res

    @staticmethod
    def to_zero(res):
        return None

    @staticmethod
    def remove_negative_values(res):
        return res.clamp(min=0)

    @staticmethod
    def to_abs(res):
        return res.abs()


class Curvature:
    HESSIAN = "hessian"
    GGN = "ggn"
    PCH_ABS = "pch-abs"
    PCH_CLIP = "pch-clip"

    CHOICES = [
        HESSIAN,
        GGN,
        PCH_CLIP,
        PCH_ABS,
    ]

    REQUIRE_PSD_LOSS_HESSIAN = {
        HESSIAN: False,
        GGN: True,
        PCH_ABS: True,
        PCH_CLIP: True,
    }

    REQUIRE_RESIDUAL = {
        HESSIAN: True,
        GGN: False,
        PCH_ABS: True,
        PCH_CLIP: True,
    }

    RESIDUAL_MODS = {
        HESSIAN: ResidualModifications.nothing,
        GGN: ResidualModifications.to_zero,
        PCH_ABS: ResidualModifications.to_abs,
        PCH_CLIP: ResidualModifications.remove_negative_values,
    }

    @classmethod
    def __check_exists(cls, which):
        if which not in cls.CHOICES:
            raise AttributeError(
                "Unknown curvature: {}. Expecting one of {}".format(which, cls.CHOICES)
            )

    @classmethod
    def require_residual(cls, curv_type):
        cls.__check_exists(curv_type)
        return cls.REQUIRE_RESIDUAL[curv_type]

    @classmethod
    def is_pch(cls, curv_type):
        """Is `curv_type` one of the PCHs proposed by Chen et al."""
        cls.__check_exists(curv_type)
        PCH = [cls.PCH_ABS, cls.PCH_CLIP]
        return curv_type in PCH

    @classmethod
    def modify_residual(cls, residual, curv_type):
        # None if zero or curvature neglects 2nd-order module effects
        if residual is None:
            return None
        else:
            cls.__check_exists(curv_type)
            return cls.RESIDUAL_MODS[curv_type](residual)

    @classmethod
    def check_loss_hessian(cls, loss_hessian_is_psd, curv_type):
        cls.__check_exists(curv_type)

        require_psd = cls.REQUIRE_PSD_LOSS_HESSIAN[curv_type]

        if require_psd and not loss_hessian_is_psd:
            raise ValueError(
                "Loss Hessian PSD = {}, but {} requires PSD = {}".format(
                    loss_hessian_is_psd, curv_type, require_psd
                )
            )

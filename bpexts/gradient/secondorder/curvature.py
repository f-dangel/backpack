class ResidualModifications():
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


class Curvature():
    HESSIAN = 'hessian'
    GGN = 'ggn'
    PCH_ABS = 'pch-abs'
    PCH_CLIP = 'pch-clip'

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
    def check_exists(cls, which):
        if not which in cls.CHOICES:
            raise AttributeError(
                "Unknown curvature matrix: {}.\n Expecting one of {}".format(
                    which, cls.CURVATURE_CHOICES))

    @classmethod
    def modify_residual(cls, residual, which):
        # None if zero or curvature neglects 2nd-order module effects
        cls.check_exists(which)
        if residual is None:
            return None
        else:
            return cls.RESIDUAL_MODS[which](residual)

    @classmethod
    def check_loss_hessian(cls, which, loss_hessian_is_psd):
        require_psd = cls.REQUIRE_PSD_LOSS_HESSIAN[which]
        if require_psd and not loss_hessian_is_psd:
            raise ValueError(
                'Loss Hessian PSD = {}, but {} requires PSD = {}'.format(
                    loss_hessian_is_psd, which, require_psd))

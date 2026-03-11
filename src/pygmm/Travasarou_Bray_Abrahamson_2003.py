""" Travasarou, Bray, and Abrahamson (2006) Arias Intensity model."""

import numpy as np

from . import model

__author__ = "James Dea"


class TravasarouBrayAbrahamson2003(model.Model):
    """Travasarou, Bray, and Abrahamson (2006, :cite:`Travasarou2006`) Arias Intensity model.

    Parameters
    ----------
    scenario : :class:`pygmm.model.Scenario`
        earthquake scenario

    """

    NAME = "Travasarou Bray Abrahamson (2003)"
    ABBREV = "TBA06"

    PARAMS = [
        model.NumericParameter("mag", True, 4.7, 7.6),
        model.NumericParameter("dist_rup", True, 0, 1000),
        # FIXME add a new parameter: site_class
        model.CategoricalParameter("site_cond", True, ["B","C","D"]),
        model.CategoricalParameter("mechanism", True, ["SS","NS","RS"])
    ]

    def __init__(self, scenario:model.Scenario):
        super().__init__(scenario)


        # total druation
        self._ln_IA = self._calc_ln_Arias_Intensity()
        self._inter_event_error = self._calc_inter_event_error()
        self._intra_event_error = self._calc_intra_event_error()
        self._total_error = np.sqrt(self._inter_event_error**2+
                                    self._intra_event_error**2)


    @property
    def Arias_Intensity(self):
        return np.exp(self._ln_IA)

    @property
    def std_inter_event_err(self):
        return self._inter_event_error
    
    @property
    def std_intra_event_err(self):
        return self._intra_event_error
    
    @property
    def std_err(self):
        return self._total_error
    
    @property
    def plus_1_sd(self):
        
        return np.exp(self._ln_IA + self._total_error)

    @property
    def minus_1_sd(self):
        
        return np.exp(self._ln_IA - self._total_error)

    def _calc_ln_Arias_Intensity(self) -> float:
        # scenario
        s = self._scenario

        c_1 = 2.800
        c_2 = -1.981
        c_3 = 20.72
        c_4 = -1.703
        h = 8.78
        S_11 = 0.454
        S_12 = 0.101
        S_21 = 0.479
        S_22 = 0.334
        f_1 = -0.166
        f_2 = 0.512
        
        # Site indicators
        S_C = 1 if s.site_cond == 'C' else 0
        S_D = 1 if s.site_cond == 'D' else 0

        F_N = 1 if s.mechanism == "NS" else 0
        F_R = 1 if s.mechanism == "RS" else 0

        ln_Ia = (
            c_1
            + c_2 * (s.mag - 6)
            + c_3 * np.log(s.mag/6)
            + c_4 * np.log(np.sqrt(s.dist_rup**2 + h**2))
            + (S_11 + S_12 * (s.mag - 6)) * S_C
            + (S_21 + S_22 * (s.mag - 6)) * S_D
            + f_1 * F_N
            + f_2 * F_R
        )

        return ln_Ia
    
    def _calc_inter_event_error(self):

        s = self._scenario

        if s.mag <= 4.7:
            tau = 0.611
        elif s.mag >= 7.6:
            tau = 0.475
        else:
            tau = 0.611-0.047*(s.mag-4.7)

        return tau

    def _calc_intra_event_error(self):
        
        s = self._scenario

        Ia = np.exp(self._calc_ln_Arias_Intensity())

        if s.site_cond == 'B':
            sigma1 = 1.18
            sigma2 = 0.94
        elif s.site_cond == 'C':
            sigma1 = 1.17
            sigma2 = 0.93
        elif s.site_cond == 'D':
            sigma1 = 0.96
            sigma2 = 0.73
        else:
            raise ValueError("site_class must be 'B', 'C', or 'D'")
        
        if Ia <= 0.013:
            sigma = sigma1
        elif 0.013 < Ia < 0.125:
            sigma = sigma1 - 0.106 * (np.log(Ia) - np.log(0.013))
        else:
            sigma = sigma2

        return sigma
from .penalty_functions import *
from .univariate_functions import *

univariates_P4 = {
    'Quad': Quadratic,
    'FourThirds': FourThirds,
    'Cosh': Cosh,
}

univariates_P5_P6_P7 = {
    'Exp': Exp,
    'LogExp': LogExp,
    'LogQuad_1': LogQuad_1,
    'LogQuad_2': LogQuad_2,
    'HyperExp': HyperExp,
    'HyperQuad': HyperQuad,
    'DualLogQuad': DualLogQuad,
    'CubicQuad': CubicQuad,
    'ExpQuad': ExpQuad,
    'LogBarrierQuad': LogBarrierQuad,
    'HyperBarrierQuad': HyperBarrierQuad,
    'HyperLogQuad': HyperLogQuad,
    'SmoothPlus': SmoothPlus,
    'NNSmoothPlus': NNSmoothPlus,
    'ExpSmoothPlus': ExpSmoothPlus,
}

univariates_P8 = {
    'LogExp': LogExp,
    'LogQuad_1': LogQuad_1,
    'LogQuad_2': LogQuad_2,
    'HyperExp': HyperExp,
    'HyperQuad': HyperQuad,
    'DualLogQuad': DualLogQuad,
    'CubicQuad': CubicQuad,
    'ExpQuad': ExpQuad,
    'LogBarrierQuad': LogBarrierQuad,
    'HyperBarrierQuad': HyperBarrierQuad,
    'HyperLogQuad': HyperLogQuad,
}

univariates_P9 = {
    'SmoothPlus': SmoothPlus,
    'NNSmoothPlus': NNSmoothPlus,
    'ExpSmoothPlus': ExpSmoothPlus,
}

combinations = {
    'PHRQuad': PHRQuad,
    'P1': P1,
    'P2': P2,
    'P3': P3,
    'P4': (P4, univariates_P4),
    'P5': (P5, univariates_P5_P6_P7),
    'P6': (P6, univariates_P5_P6_P7),
    'P7': (P7, univariates_P5_P6_P7),
    'P8': (P8, univariates_P8),
    'P9': (P9, univariates_P9),
}

all_penalties = {}
for p_name, penalty in combinations.items():
    if isinstance(penalty, tuple):
        for θ_name, θ in penalty[1].items():
            all_penalties['_'.join([p_name, θ_name])] = penalty[0](θ())
    else:
        all_penalties[p_name] = penalty

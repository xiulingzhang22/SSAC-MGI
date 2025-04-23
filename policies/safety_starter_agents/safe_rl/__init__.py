from tensorflow.python.util import deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from .pg.algos import ppo, ppo_lagrangian, trpo, trpo_lagrangian, cpo
from .sac.sac import sac
from .pg import trust_region
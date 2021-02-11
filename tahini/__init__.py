from pkg_resources import get_distribution as _get_distribution

__author__ = 'tahini-dev'
__version__ = _get_distribution('tahini').version

from . import (
    core,
    testing,
    plot,
)

from .core import (
    Graph,
)

from .factory import (
    get_path,
    get_star,
    get_complete,
)

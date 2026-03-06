from .wordcloud import (WordCloud, STOPWORDS, random_color_func,
                        get_single_color_func)
from .color_from_image import ImageColorGenerator
from .group_placement import GroupAwarePlacer

__all__ = ['WordCloud', 'STOPWORDS', 'random_color_func',
           'get_single_color_func', 'ImageColorGenerator',
           'GroupAwarePlacer', '__version__']

from ._version import __version__

import colorsys
import numpy as np

from random import Random
from PIL import ImageFont
from PIL import ImageColor


class ImageColorGenerator(object):
    """Color generator based on a color image.

    Generates colors based on an RGB image. A word will be colored using
    the mean color of the enclosing rectangle in the color image.

    After construction, the object acts as a callable that can be passed as
    color_func to the word cloud constructor or to the recolor method.

    Parameters
    ----------
    image : nd-array, shape (height, width, 3)
        Image to use to generate word colors. Alpha channels are ignored.
        This should be the same size as the canvas. for the wordcloud.
    default_color : tuple or None, default=None
        Fallback colour to use if the canvas is larger than the image,
        in the format (r, g, b). If None, raise ValueError instead.
    """
    # returns the average color of the image in that region
    def __init__(self, image, default_color=None):
        if image.ndim not in [2, 3]:
            raise ValueError("ImageColorGenerator needs an image with ndim 2 or"
                             " 3, got %d" % image.ndim)
        if image.ndim == 3 and image.shape[2] not in [3, 4]:
            raise ValueError("A color image needs to have 3 or 4 channels, got %d"
                             % image.shape[2])
        self.image = image
        self.default_color = default_color

    def __call__(self, word, font_size, font_path, position, orientation, **kwargs):
        """Generate a color for a given word using a fixed image."""
        # get the font to get the box size
        font = ImageFont.truetype(font_path, font_size)
        transposed_font = ImageFont.TransposedFont(font,
                                                   orientation=orientation)
        # get size of resulting text
        box_size = transposed_font.getsize(word)
        x = position[0]
        y = position[1]
        # cut out patch under word box
        patch = self.image[x:x + box_size[0], y:y + box_size[1]]
        if patch.ndim == 3:
            # drop alpha channel if any
            patch = patch[:, :, :3]
        if patch.ndim == 2:
            raise NotImplementedError("Gray-scale images TODO")
        # check if the text is within the bounds of the image
        reshape = patch.reshape(-1, 3)
        if not reshape.any():
            if self.default_color is None:
                raise ValueError('ImageColorGenerator is smaller than the canvas')
            return "rgb(%d, %d, %d)" % tuple(self.default_color)
        color = np.mean(reshape, axis=0)
        return "rgb(%d, %d, %d)" % tuple(color)


def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    """Random hue color generation.

    Default coloring method. This just picks a random hue with value 80% and
    lumination 50%.

    Parameters
    ----------
    word, font_size, position, orientation  : ignored.

    random_state : random.Random object or None, (default=None)
        If a random object is given, this is used for generating random
        numbers.

    """
    if random_state is None:
        random_state = Random()
    return "hsl(%d, 80%%, 50%%)" % random_state.randint(0, 255)


class colormap_color_func(object):
    """Color func created from matplotlib colormap.

    Parameters
    ----------
    colormap : string or matplotlib colormap
        Colormap to sample from

    Example
    -------
    >>> WordCloud(color_func=colormap_color_func("magma"))

    """
    def __init__(self, colormap):
        import matplotlib.pyplot as plt
        self.colormap = plt.cm.get_cmap(colormap)

    def __call__(self, word, font_size, position, orientation,
                 random_state=None, **kwargs):
        if random_state is None:
            random_state = Random()
        r, g, b, _ = np.maximum(0, 255 * np.array(self.colormap(
            random_state.uniform(0, 1))))
        return "rgb({:.0f}, {:.0f}, {:.0f})".format(r, g, b)


def get_single_color_func(color):
    """Create a color function which returns a single hue and saturation with.
    different values (HSV). Accepted values are color strings as usable by
    PIL/Pillow.

    >>> color_func1 = get_single_color_func('deepskyblue')
    >>> color_func2 = get_single_color_func('#00b4d2')
    """
    old_r, old_g, old_b = ImageColor.getrgb(color)
    rgb_max = 255.
    h, s, v = colorsys.rgb_to_hsv(old_r / rgb_max, old_g / rgb_max,
                                  old_b / rgb_max)

    def single_color_func(word=None, font_size=None, position=None,
                          orientation=None, font_path=None, random_state=None):
        """Random color generation.

        Additional coloring method. It picks a random value with hue and
        saturation based on the color given to the generating function.

        Parameters
        ----------
        word, font_size, position, orientation  : ignored.

        random_state : random.Random object or None, (default=None)
          If a random object is given, this is used for generating random
          numbers.

        """
        if random_state is None:
            random_state = Random()
        r, g, b = colorsys.hsv_to_rgb(h, s, random_state.uniform(0.2, 1))
        return 'rgb({:.0f}, {:.0f}, {:.0f})'.format(r * rgb_max, g * rgb_max,
                                                    b * rgb_max)
    return single_color_func

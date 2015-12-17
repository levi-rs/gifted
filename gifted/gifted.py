# -*- coding: utf-8 -*-

""" Module gifted

Provides functionality for reading and writing animated GIF images.
Use write_gif to write a series of numpy arrays or PIL images as an
animated GIF. Use read_gif to read an animated gif as a series of numpy
arrays.

Note that since July 2004, all patents on the LZW compression patent have
expired. Therefore the GIF format may now be used freely.

Acknowledgements
----------------

Many thanks to Ant1 for:
* noting the use of "palette=PIL.Image.ADAPTIVE", which significantly
  improves the results.
* the modifications to save each image with its own palette, or optionally
  the global palette (if its the same).

Many thanks to Marius van Voorden for porting the NeuQuant quantization
algorithm of Anthony Dekker to Python (See the NeuQuant class for its
license).

Many thanks to Alex Robinson for implementing the concept of subrectangles,
which (depening on image content) can give a very significant reduction in
file size.

This code is based on gifmaker (in the scripts folder of the source
distribution of PIL)


Usefull links
-------------
  * http://tronche.com/computer-graphics/gif/
  * http://en.wikipedia.org/wiki/Graphics_Interchange_Format
  * http://www.w3.org/Graphics/GIF/spec-gif89a.txt

"""

import os
from fnmatch import fnmatch

import numpy as np
import PIL
from PIL import Image

from gifted.gif_writer import GIFWriter


def get_cKDTree():
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    return cKDTree


# getheader gives a 87a header and a color palette (two elements in a list).
# getdata()[0] gives the Image Descriptor up to (including) "LZW min code size".
# getdatas()[1:] is the image data itself in chuncks of 256 bytes (well
# technically the first byte says how many bytes follow, after which that
# amount (max 255) follows).

def check_images(images):
    """ check_images(images)
    Check numpy images and correct intensity range etc.
    The same for all movie formats.
    """
    # Init results
    images2 = []

    for img in images:
        if PIL and isinstance(img, PIL.Image.Image):
            # We assume PIL images are allright
            images2.append(img)

        elif np and isinstance(img, np.ndarray):
            # Check and convert dtype
            if img.dtype == np.uint8:
                images2.append(img)  # Ok
            elif img.dtype in [np.float32, np.float64]:
                img = img.copy()
                img[img < 0] = 0
                img[img > 1] = 1
                img *= 255
                images2.append(img.astype(np.uint8))
            else:
                img = img.astype(np.uint8)
                images2.append(img)
            # Check size
            if img.ndim == 2:
                pass  # ok
            elif img.ndim == 3:
                if img.shape[2] not in [3, 4]:
                    raise ValueError('This array can not represent an image.')
            else:
                raise ValueError('This array can not represent an image.')
        else:
            raise ValueError('Invalid image type: ' + str(type(img)))

    # Done
    return images2


# Exposed functions
def write_gif(filename, images, duration=0.1,
              repeat=True, dither=False, nq=0, sub_rectangles=True, dispose=None):
    """ write_gif(filename, images, duration=0.1, repeat=True, dither=False,
                    nq=0, sub_rectangles=True, dispose=None)

    Write an animated gif from the specified images.

    Parameters
    ----------
    filename : string
        The name of the file to write the image to.
    images : list
        Should be a list consisting of PIL images or numpy arrays.
        The latter should be between 0 and 255 for integer types, and
        between 0 and 1 for float types.
    duration : scalar or list of scalars
        The duration for all frames, or (if a list) for each frame.
    repeat : bool or integer
        The amount of loops. If True, loops infinitetely.
    dither : bool
        Whether to apply dithering
    nq : integer
        If nonzero, applies the NeuQuant quantization algorithm to create
        the color palette. This algorithm is superior, but slower than
        the standard PIL algorithm. The value of nq is the quality
        parameter. 1 represents the best quality. 10 is in general a
        good tradeoff between quality and speed. When using this option,
        better results are usually obtained when sub_rectangles is False.
    sub_rectangles : False, True, or a list of 2-element tuples
        Whether to use sub-rectangles. If True, the minimal rectangle that
        is required to update each frame is automatically detected. This
        can give significant reductions in file size, particularly if only
        a part of the image changes. One can also give a list of x-y
        coordinates if you want to do the cropping yourself. The default
        is True.
    dispose : int
        How to dispose each frame. 1 means that each frame is to be left
        in place. 2 means the background color should be restored after
        each frame. 3 means the decoder should restore the previous frame.
        If sub_rectangles==False, the default is 2, otherwise it is 1.

    """

    # Check PIL
    if PIL is None:
        raise RuntimeError("Need PIL to write animated gif files.")

    # Check images
    images = check_images(images)

    # Instantiate writer object
    gif_writer = GIFWriter()
    gif_writer.transparency = False  # init transparency flag used in GIFWriter functions

    # Check loops
    if repeat is False:
        loops = 1
    elif repeat is True:
        loops = 0  # zero means infinite
    else:
        loops = int(repeat)

    # Check duration
    if hasattr(duration, '__len__'):
        if len(duration) == len(images):
            duration = [d for d in duration]
        else:
            raise ValueError("len(duration) doesn't match amount of images.")
    else:
        duration = [duration for im in images]

    # Check subrectangles
    if sub_rectangles:
        images, xy = gif_writer.handle_sub_rectangles(images, sub_rectangles)
        default_dispose = 1  # Leave image in place
    else:
        # Normal mode
        xy = [(0, 0) for im in images]
        default_dispose = 2  # Restore to background color.

    # Check dispose
    if dispose is None:
        dispose = default_dispose
    if hasattr(dispose, '__len__'):
        if len(dispose) != len(images):
            raise ValueError("len(xy) doesn't match amount of images.")
    else:
        dispose = [dispose for im in images]

    # Make images in a format that we can write easy
    images = gif_writer.convert_images_to_PIL(images, dither, nq)

    # Write
    with open(filename, 'wb') as file_:
        gif_writer.write_gif_to_file(file_, images, duration, loops, xy, dispose)


def read_gif(filename, as_numpy=True):
    """ read_gif(filename, as_numpy=True)

    Read images from an animated GIF file.  Returns a list of numpy
    arrays, or, if as_numpy is false, a list if PIL images.

    """

    # Check PIL
    if PIL is None:
        raise RuntimeError("Need PIL to read animated gif files.")

    # Check Numpy
    if np is None:
        raise RuntimeError("Need Numpy to read animated gif files.")

    # Check whether it exists
    if not os.path.isfile(filename):
        raise IOError('File not found: '+str(filename))

    # Load file using PIL
    pil_image = PIL.Image.open(filename)
    pil_image.seek(0)

    # Read all images inside
    images = []
    try:
        while True:
            # Get image as numpy array
            tmp = pil_image.convert()  # Make without palette
            array_ = np.asarray(tmp)
            if len(array_.shape) == 0:
                raise MemoryError("Too little memory to convert PIL image to array")
            # Store, and next
            images.append(array_)
            pil_image.seek(pil_image.tell()+1)
    except EOFError:
        pass

    # Convert to normal PIL images if needed
    if not as_numpy:
        images2 = images
        images = []
        for image in images2:
            tmp = PIL.Image.fromarray(image)
            images.append(tmp)

    # Done
    return images


def load_images(image_directory, extension, prefix=None):
    """
    Locates image files in image_directory with the specified extension and/or
    prefix, and loads them into memory as PIL/Pillow objects

    :param image_directory: string
    :param extension: string
    :param prefix: string
    :returns: List of PIL Image objects
    """
    exl = extension.lower()
    exu = extension.upper()

    # List everything in dir:
    all_files = os.listdir(image_directory)

    # Prune out unwanted extension types
    images = [i for i in all_files if fnmatch(i, "*." + exl) or fnmatch(i, "*." + exu)]

    # Prune out unwanted prefix types
    if prefix:
        images = [i for i in images if fnmatch(i, prefix + "*")]

    # Sort to maintain order during GIF creation
    images.sort()

    return [Image.open(os.path.join(image_directory, i)) for i in images]

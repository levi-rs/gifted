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
from PIL.GifImagePlugin import getheader, getdata

from gifted.neuquant import NeuQuant


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


def int_to_bin(i):
    """ Integer to two bytes """

    # make string (little endian)
    return i.to_bytes(2, byteorder='little')


class GifWriter(object):
    """ GifWriter()

    Class that contains methods for helping write the animated GIF file.

    """

    def __init__(self):
        self.transparency = None

    @staticmethod
    def get_header_anim(img):
        """ get_header_anim(img)

        Get animation header. To replace PILs getheader()[0]

        """
        header = b'GIF89a'
        header += int_to_bin(img.size[0])
        header += int_to_bin(img.size[1])
        header += b'\x87\x00\x00'

        return header

    @staticmethod
    def get_image_descriptor(img, coords=None):
        """ get_image_descriptor(img, coords=None)

        Used for the local color table properties per image.
        Otherwise global color table applies to all frames irrespective of
        whether additional colors comes in play that require a redefined
        palette. Still a maximum of 256 color per frame, obviously.

        Written by Ant1 on 2010-08-22
        Modified by Alex Robinson in Janurari 2011 to implement subrectangles.

        """
        # Defaule use full image and place at upper left
        if coords is None:
            coords = (0, 0)

        # Image separator
        descriptor = b'\x2C'

        # Image position and size
        descriptor += int_to_bin(coords[0])    # Left position
        descriptor += int_to_bin(coords[1])    # Top position
        descriptor += int_to_bin(img.size[0])  # image width
        descriptor += int_to_bin(img.size[1])  # image height

        # packed field: local color table flag1, interlace0, sorted table0,
        # reserved00, lct size111=7=2^(7+1)=256.
        descriptor += b'\x87'

        # LZW minimum size code now comes later, begining of [image data] blocks
        return descriptor

    @staticmethod
    def get_app_ext(loops=float('inf')):
        """ get_app_ext(loops=float('inf'))

        Application extention. This part specifies the amount of loops.
        If loops is 0 or inf, it goes on infinitely.

        """

        if loops == 0 or loops == float('inf'):
            loops = 2**16-1
            # bb = ""  # application extension should not be used
            #          # (the extension interprets zero loops
            #          # to mean an infinite number of loops)
            #          # Mmm, does not seem to work

        ext = b"\x21\xFF\x0B"  # application extension
        ext += b"NETSCAPE2.0"
        ext += b"\x03\x01"
        ext += int_to_bin(loops)
        ext += b'\x00'  # end

        return ext

    @staticmethod
    def get_graphics_control_ext(
            duration=0.1, dispose=2, transparent_flag=0, transparency_index=0):
        """ get_graphics_control_ext(duration=0.1, dispose=2)

        Graphics Control Extension. A sort of header at the start of
        each image. Specifies duration and transparancy.

        Dispose
        -------
          * 0 - No disposal specified.
          * 1 - Do not dispose. The graphic is to be left in place.
          * 2 -	Restore to background color. The area used by the graphic
            must be restored to the background color.
          * 3 -	Restore to previous. The decoder is required to restore the
            area overwritten by the graphic with what was there prior to
            rendering the graphic.
          * 4-7 -To be defined.

        """

        ext = b'\x21\xF9\x04'
        ext += bytes([((dispose & 3) << 2) | (transparent_flag & 1)])  # low bit 1 == transparency,
        # 2nd bit 1 == user input , next 3 bits, the low two of which are used,
        # are dispose.
        ext += int_to_bin(int(duration * 100))  # in 100th of seconds
        ext += bytes([transparency_index])
        ext += b'\x00'  # end

        return ext

    def handle_sub_rectangles(self, images, sub_rectangles):
        """ handle_sub_rectangles(images)

        Handle the sub-rectangle stuff. If the rectangles are given by the
        user, the values are checked. Otherwise the subrectangles are
        calculated automatically.

        """
        if isinstance(sub_rectangles, (tuple, list)):
            # xy given directly

            # Check xy
            sub_recs = sub_rectangles
            if sub_recs is None:
                sub_recs = (0, 0)
            if hasattr(sub_recs, '__len__'):
                if len(sub_recs) == len(images):
                    sub_recs = [xxyy for xxyy in sub_recs]
                else:
                    raise ValueError("len(sub_recs) doesn't match amount of images.")
            else:
                sub_recs = [sub_recs for image in images]
            sub_recs[0] = (0, 0)

        else:
            # Calculate xy using some basic image processing

            # Check Numpy
            if np is None:
                raise RuntimeError("Need Numpy to use auto-sub_rectangles.")

            # First make numpy arrays if required
            for i in range(len(images)):
                image = images[i]
                if isinstance(image, Image.Image):
                    tmp = image.convert()  # Make without palette
                    array_ = np.asarray(tmp)
                    if len(array_.shape) == 0:
                        raise MemoryError("Too little memory to convert PIL image to array")
                    images[i] = array_

            # Determine the sub rectangles
            images, sub_rec = self.get_sub_rectangles(images)

        # Done
        return images, sub_rec

    @staticmethod
    def get_sub_rectangles(images):
        """ get_sub_rectangles(images)

        Calculate the minimal rectangles that need updating each frame.
        Returns a two-element tuple containing the cropped images and a
        list of x-y positions.

        Calculating the subrectangles takes extra time, obviously. However,
        if the image sizes were reduced, the actual writing of the GIF
        goes faster. In some cases applying this method produces a GIF faster.

        """

        # Check image count
        if len(images) < 2:
            return images, [(0, 0) for i in images]

        # We need numpy
        if np is None:
            raise RuntimeError("Need Numpy to calculate sub-rectangles. ")

        # Prepare
        ims2 = [images[0]]
        coords = [(0, 0)]

        # Iterate over images
        prev = images[0]
        for image in images[1:]:

            # Get difference, sum over colors
            diff = np.abs(image - prev)
            if diff.ndim == 3:
                diff = diff.sum(2)
            # Get begin and end for both dimensions
            X = np.argwhere(diff.sum(0))
            Y = np.argwhere(diff.sum(1))
            # Get rect coordinates
            if X.size and Y.size:
                x0, x1 = int(X[0][0]), int(X[-1][0]+1)
                y0, y1 = int(Y[0][0]), int(Y[-1][0]+1)
            else:  # No change ... make it minimal
                x0, x1 = 0, 2
                y0, y1 = 0, 2

            # Cut out and store
            im2 = image[y0:y1, x0:x1]
            prev = image
            ims2.append(im2)
            coords.append((x0, y0))

        return ims2, coords

    def convert_images_to_PIL(self, images, dither, nq=0):
        """ convert_images_to_PIL(images, nq=0)

        Convert images to Paletted PIL images, which can then be
        written to a single animaged GIF.

        """

        # Convert to PIL images
        images2 = []
        for image in images:
            if isinstance(image, Image.Image):
                images2.append(image)
            elif np and isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[2] == 3:
                    image = Image.fromarray(image, 'RGB')
                elif image.ndim == 3 and image.shape[2] == 4:
                    # image = Image.fromarray(image[:,:,:3],'RGB')
                    self.transparency = True
                    image = Image.fromarray(image[:, :, :4], 'RGBA')
                elif image.ndim == 2:
                    image = Image.fromarray(image, 'L')
                images2.append(image)

        # Convert to paletted PIL images
        images, images2 = images2, []
        if nq >= 1:
            # NeuQuant algorithm
            for image in images:
                image = image.convert("RGBA")  # NQ assumes RGBA
                # Learn colors from image
                nq_instance = NeuQuant(image, int(nq))
                if dither:
                    image = image.convert("RGB").quantize(
                        palette=nq_instance.paletteImage(), colors=255)
                else:
                    # Use to quantize the image itself
                    image = nq_instance.quantize(image, colors=255)

                # since NQ assumes transparency
                self.transparency = True
                if self.transparency:
                    alpha = image.split()[3]
                    mask = Image.eval(alpha, lambda a: 255 if a <= 128 else 0)
                    image.paste(255, mask=mask)
                images2.append(image)
        else:
            # for index,image in enumerate(images):
            for i in range(len(images)):
                image = images[i].convert('RGB').convert(
                    'P',
                    palette=Image.ADAPTIVE,  # Adaptive PIL algorithm
                    dither=dither,
                    colors=255
                )
                if self.transparency:
                    alpha = images[i].split()[3]
                    mask = Image.eval(alpha, lambda a: 255 if a <= 128 else 0)
                    image.paste(255, mask=mask)
                images2.append(image)

        # Done
        return images2

    def write_gif_to_file(self, file_, images, durations, loops, xys, disposes):
        """ write_gif_to_file(file_, images, durations, loops, xys, disposes)

        Given a set of images writes the bytes to the specified stream.

        """

        # Obtain palette for all images and count each occurance
        palettes, occur = [], []
        for image in images:
            palettes.append(getheader(image)[0][3])

        for palette in palettes:
            occur.append(palettes.count(palette))

        # Select most-used palette as the global one (or first in case no max)
        global_palette = palettes[occur.index(max(occur))]

        # Init
        frames = 0
        first_frame = True

        for image, palette in zip(images, palettes):

            if first_frame:
                # Write header

                # Gather info
                header = self.get_header_anim(image)
                appext = self.get_app_ext(loops)

                # Write
                file_.write(header)
                file_.write(global_palette)
                file_.write(appext)

                # Next frame is not the first
                first_frame = False

            if True:
                # Write palette and image data

                # Gather info
                data = getdata(image)
                imdes, data = data[0], data[1:]

                transparent_flag = 1 if self.transparency else 0

                graphext = self.get_graphics_control_ext(
                    durations[frames],
                    disposes[frames],
                    transparent_flag=transparent_flag,
                    transparency_index=255
                )

                # Make image descriptor suitable for using 256 local color palette
                lid = self.get_image_descriptor(image, xys[frames])

                # Write local header
                if (palette != global_palette) or (disposes[frames] != 2):
                    # Use local color palette
                    file_.write(graphext)
                    file_.write(lid)      # write suitable image descriptor
                    file_.write(palette)  # write local color table
                    file_.write(b'\x08')  # LZW minimum size code
                else:
                    # Use global color palette
                    file_.write(graphext)
                    file_.write(imdes)  # write suitable image descriptor

                # Write image data
                for datum in data:
                    file_.write(datum)

            # Prepare for next round
            frames = frames + 1

        file_.write(b';')  # end gif

        return frames


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
    gif_writer = GifWriter()
    gif_writer.transparency = False  # init transparency flag used in GifWriter functions

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

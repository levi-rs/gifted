from PIL import Image
from PIL.GifImagePlugin import getheader, getdata
import numpy as np

from gifted.neuquant import NeuQuant


def int_to_bin(i):
    """ Integer to two bytes """

    # make string (little endian)
    return i.to_bytes(2, byteorder='little')


class GIFWriter(object):
    """ GIFWriter()

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

import os
import argparse

from gifted.libgif import load_images, write_gif


# Strings
PNG = 'PNG'
png = 'png'
JPG = 'JPG'
jpg = 'jpg'
GIF = 'GIF'
gif = 'gif'

OUTPUT_FILE = "output.gif"
DEFAULT_DURATION = 0.2


def get_args():
    """
    Parses command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d', '--directory',
        type=str, required=True,
        help="Folder to load images from"
    )
    parser.add_argument(
        '-e', '--extension',
        type=str, default=PNG, choices=[PNG, png, JPG, jpg, GIF, gif],
        help="Image extension type"
    )
    parser.add_argument(
        '-o', '--output-file',
        type=str, default=OUTPUT_FILE,
        help='The name of the output file. Defaults to {0}'.format(OUTPUT_FILE)
    )
    parser.add_argument(
        '--duration',
        type=float, default=DEFAULT_DURATION,
        help="Duration between frames. Defaults to {0}".format(DEFAULT_DURATION)
    )
    parser.add_argument(
        '--dither',
        type=bool, default=False, choices=[True, False],
        help="Use dither when creating GIF"
    )

    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.isdir(args.directory):
        raise ValueError("Cannot find directory {0}".format(args.directory))

    imgs = load_images(args.directory, args.extension)

    write_gif(args.output_file, imgs, args.duration, args.dither)


if __name__ == "__main__":
    main()

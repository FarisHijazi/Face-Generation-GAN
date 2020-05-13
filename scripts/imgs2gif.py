from PIL import Image
import glob
import os
from argparse import ArgumentParser

parser = ArgumentParser('Convert to a group of images to a GIF')
parser.add_argument('input', metavar="DIRECTORY", default='.')
parser.add_argument('-o', '--output', metavar="OUTPUT_GIF", default=None)
parser.add_argument('-d', '--duration', default=150,
                    help='Duration between frames (in ms)')
parser.add_argument('--scale', metavar='SCALE_FACTOR', default=0.3)
args = parser.parse_args()


if args.output is None:
    args.output = 'gif.gif'

if len(os.path.split(args.output)) <= 2:
    args.output = os.path.join(args.input, args.output)


# Create the frames
frames = []
imgs = glob.glob(os.path.join(args.input, "*.png"))
for img in imgs:
    img = Image.open(img)

    w, h = int(
        img.size[0]*args.scale), int(img.size[1]*args.scale)

    frames.append(img.resize((w, h), Image.ANTIALIAS))

print(f'{len(frames)} images found')

if len(frames):
    # Save into a GIF file that loops forever
    frames[0].save(args.output, format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=args.duration, loop=0)
    
    print('saved image in:', args.output)

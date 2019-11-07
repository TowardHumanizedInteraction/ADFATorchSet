"""Main

The file profides simple access to the library using terminal arguments in order
to download and generate the dataset for the ADFA project.
"""
import argparse

from adfatorchset.download import download_ravdess
from adfatorchset.metadata import extract_metadata

from adfatorchset.download import download_3DDFA
download_3DDFA( )

from adfatorchset.preprocess import preprocess_3ddfa
from adfatorchset.extract import extract_dataset
from adfatorchset.dfa import extract_3ddfa

parser = argparse.ArgumentParser( )
parser.add_argument(
    '-i', '--input',
    help     = 'Input folder to find the video dataset',
    type     = str
)
parser.add_argument(
    '-o', '--output',
    help     = 'Output folder to save the extracted content in',
    type     = str
)
parser.add_argument(
    '-f', '--fps',
    help    = 'Chosen frame rate for the frame extraction',
    type    = int,
    default = 30
)
parser.add_argument(
    '-s', '--sr',
    help    = 'Chosen sample rate for the audio extraction',
    type    = int,
    default = 16000
)
parser.add_argument(
    '-b', '--batch_size',
    help    = 'Batch size for facial reconstruction',
    type    = int,
    default = 32
)
parser.add_argument(
    '-d', '--extract_dataset',
    help   = 'Extract frames and audio from dataset',
    action = 'store_true'
)
parser.add_argument(
    '-c', '--cuda',
    help   = 'Choose to perform 3D facial reconstruction on gpu',
    action = 'store_true'
)
parser.add_argument(
    '-v', '--verbose',
    help   = 'Choose to output ffmpeg logs',
    action = 'store_true'
)
parser.add_argument(
    '-r', '--ravdess',
    help   = 'Download RAVDESS dataset',
    action = 'store_true'
)
parser.add_argument(
    '-p', '--preprocess',
    help   = 'Preprocess faces to project into frontal facial plane',
    action = 'store_true'
)
parser.add_argument(
    '-x', '--no_extraction',
    help   = 'No extraction needed (maybe to apply preprocess only)',
    action = 'store_true'
)
args = parser.parse_args( )

if args.ravdess:
    download_ravdess( args.input )

if not args.no_extraction:
    extract_dataset(
        args.input,
        args.output,
        fps     = args.fps,
        sr      = args.sr,
        verbose = args.verbose
    )
    extract_3ddfa(
        args.output,
        batch_size = args.batch_size,
        cuda       = args.cuda
    )
    extract_metadata( args.output )

if args.preprocess:
    preprocess_3ddfa( args.output )

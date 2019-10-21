"""Main

The file profides simple access to the library using terminal arguments in order
to download and generate the dataset for the ADFA project.
"""
if __name__ == '__main__':
    import argparse

    from adfaset.download import download_ravdess
    from adfaset.metadata import extract_metadata
    from adfaset.download import download_3DDFA
    from adfaset.extract import extract_dataset
    from adfaset.dfa import extract_3ddfa

    parser = argparse.ArgumentParser( )
    parser.add_argument(
        '-i', '--input',
        help     = 'Input folder to find the video dataset',
        type     = str,
        required = True
    )
    parser.add_argument(
        '-o', '--output',
        help     = 'Output folder to save the extracted content in',
        type     = str,
        required = True
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
    args = parser.parse_args( )

    download_3DDFA( )

    if args.ravdess:
        download_ravdess( args.input )

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

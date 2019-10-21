"""Extract

The file contains methods to extract the frames, and audio from a video ravdess
like dataset. Frames are saved to PNG format, and audio to WAV. All extraction
processes are done using ffmpeg.
"""
import os

from multiprocessing import Pool
from typing import Tuple
from tqdm import tqdm

def extract_frames( src: str, dst: str, fps: int, option: str ) -> None:
    """Extract Frames

    Parameters
    ----------
    src   : str
            Source mp4 file to extract the frames from.
    dst   : str
            Destination folder for the extracted frames.
    fps   : int
            Frame rate at which the frames are extracted from the video.
    option: str
            Option string for the ffmpeg command.
    """
    os.system( f'ffmpeg -i { src } -vf fps={ fps } { dst } -hide_banner { option }' )

def extract_audio( src: str, dst: str, sr: int, option: str ) -> None:
    """Extract Audio

    Parameters
    ----------
    src   : str
            Source mp4 file to extract the audio from.
    dst   : str
            Destination folder for the extracted audio.
    sr    : int
            Sample rate at which the audio is extracted from the video.
    option: str
            Option string for the ffmpeg command.
    """
    os.system( f'ffmpeg -i { src } -ar { sr } -ac 1 { dst } { option }' )

def extract_video( params: Tuple[ str, str, int, int, bool ] ) -> None:
    """Exract Video

    Parameters
    ----------
    params: Tuple[ str, str, int, int, bool ]
            Source, Destination, FPS, Sample Rate, Option
    """
    src, dst, fps, sr, verbose = params

    if not os.path.isdir( dst ):
        os.makedirs( dst )

    verbose = '-loglevel panic' if not verbose else ''
    frame   = os.path.join( dst, 'frame_%06d.png' )
    audio   = os.path.join( dst, 'audio.wav' )

    extract_frames( src, frame, fps, verbose )
    extract_audio( src, audio, sr, verbose )

def filter_video( params: Tuple[ str, str, int, int, bool ] ) -> bool:
    """Filter Video with Audio

    Parameters
    ----------
    params: Tuple[ str, str, int, int, bool ]
            Source, Destination, FPS, Sample Rate, Option

    Returns
    -------
    res: bool
         Wether or not the video contains audio.
    """
    return int( params[ 0 ].split( '/' )[ -1 ].split( '-' )[ 0 ] ) == 1

def extract_actor( src: str, dst: str, fps: int, sr: int, verbose: bool ) -> None:
    """Extract Actor

    Parameters
    ----------
    src    : str
             Source mp4 file to extract the data from.
    dst    : str
             Destination folder for the extracted data.
    fps    : int
             Frame rate at which the frames are extracted from the video.
    sr     : int
             Sample rate at which the audio is extracted from the video.
    verbose: bool
             Option to make ffmpeg log.
    """

    def get_args( src: str, dst: str, mp4: str, fps: int, sr: int, verbose: bool ) -> Tuple[ str, str, int, int, bool ]:
        src = os.path.join( src, mp4 )
        dst = os.path.join( dst, mp4.replace( '.mp4', '' ) )

        return src, dst, fps, sr, verbose

    with Pool( ) as pool:
        iter = list( filter( filter_video, [
            get_args( src, dst, mp4, fps, sr, verbose )
            for mp4 in os.listdir( src ) if 'mp4' in mp4
        ] ) )
        imap = pool.imap_unordered( extract_video, iter )
        pbar = tqdm(
            imap,
            total = len( iter ),
            desc  = 'Extracting Frames and Audio from Videos'
        )
        _    = list( pbar )

def extract_dataset( src: str, dst: str, fps: int = 30, sr: int = 16000, verbose: bool = False ) -> None:
    """Extract Dataset

    Parameters
    ----------
    src    : str
             Source mp4 file to extract the data from.
    dst    : str
             Destination folder for the extracted data.
    fps    : int
             default 30
             Frame rate at which the frames are extracted from the video.
    sr     : int
             default 16 kHz ( 16000 )
             Sample rate at which the audio is extracted from the video.
    verbose: bool
             default False
             Option to make ffmpeg log.
    """

    iter = [
        os.path.join( src, actor )
        for actor in os.listdir( src ) \
        if os.path.isdir( os.path.join( src, actor ) )
    ]

    for actor in tqdm( sorted( iter ), desc = 'Actor' ):
        actor_dst = os.path.join( dst, actor.split( '/' )[ -1 ] )
        extract_actor( actor, actor_dst, fps, sr, verbose )

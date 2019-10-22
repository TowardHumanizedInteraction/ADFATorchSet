"""Download

The file provides simple methods to download dependencies to the 3DDFA modle
for facial reconstruction from monocular images [1] and the RAVDESS dataset [2].

[1] https://github.com/cleardusk/3DDFA
[2] @article{
    10.1371/journal.pone.0196391,
    author = {Livingstone, Steven R. AND Russo, Frank A.},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {
        The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS):
        A dynamic, multimodal set of facial and vocal expressions in North
        American English
    },
    year = {2018},
    month = {05},
    volume = {13},
    url = {https://doi.org/10.1371/journal.pone.0196391},
    pages = {1-35},
    abstract = {
        The RAVDESS is a validated multimodal database of emotional
        speech and song. The database is gender balanced consisting of 24
        professional actors, vocalizing lexically-matched statements in a
        neutral North American accent. Speech includes calm, happy, sad, angry,
        fearful, surprise, and disgust expressions, and song contains calm,
        happy, sad, angry, and fearful emotions. Each expression is produced at
        two levels of emotional intensity, with an additional neutral
        expression. All conditions are available in face-and-voice, face-only,
        and voice-only formats. The set of 7356 recordings were each rated 10
        times on emotional validity, intensity, and genuineness. Ratings were
        provided by 247 individuals who were characteristic of untrained
        research participants from North America. A further set of 72
        participants provided test-retest data. High levels of emotional
        validity and test-retest intrarater reliability were reported. Corrected
        accuracy and composite "goodness" measures are presented to assist
        researchers in the selection of stimuli. All recordings are made freely
        available under a Creative Commons license and can be downloaded at
        https://doi.org/10.5281/zenodo.1188976.
    },
    number = {5},
    doi = {10.1371/journal.pone.0196391}
}
"""
import os

from zipfile import ZipFile
from tqdm import tqdm

"""Urls

Constants
---------
GIT_3DDFA  : str
             Url to the 3DDFA github repository
URL_3DDFA  : str
             Url to the 68 face landmarks dlib predictor.
URL_RAVDESS: str
             Base url for the RAVDESS dataset zip files.
"""
GIT_3DDFA   = 'https://github.com/cleardusk/3DDFA.git'
URL_3DDFA   = 'https://github.com/AKSHAYUBHAT/TensorFace/raw/master' + \
              '/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
URL_RAVDESS = 'https://zenodo.org/record/1188976/files'

def download_3DDFA(  ) -> None:
    """Download 3DDFA
    """

    if not os.path.isdir( './3DDFA' ):
        os.system( f'''
            git clone { GIT_3DDFA } && \
            cd ./3DDFA/utils/cython && \
            python3 setup.py build_ext -i && \
            cd ../../models &&
            wget { URL_3DDFA }
            cd ../
        ''' )

def download_ravdess( dst : str ) -> None:
    """Download RAVDESS

    Download the RAVDESS dataset video zip files, extract all, and clean.

    Parameters
    ----------
    dst: string
         Destination path for the ravdess video files.
    """

    if not os.path.isdir( dst ):
        os.makedirs( dst )

    url  = '{2}/Video_{0}_Actor_{1:02d}.zip'
    name = 'Video_{0}_Actor_{1:02d}.zip'

    pbar = tqdm( range( 24 ), desc = 'Downloading Dataset Actor [00/24]' )
    for i in pbar:
        id  = i + 1
        if os.path.isdir( os.path.join( dst, f'Actor_{id:02d}' ) ):
            continue

        pbar.set_description( f'Downloading Dataset Actor [{id:02d}/24]' )

        speech      = url.format( 'Speech', id, URL_RAVDESS )
        song        = url.format( 'Song', id, URL_RAVDESS )
        speech_path = os.path.join( dst, name.format( 'Speech', id ) )
        song_path   = os.path.join( dst, name.format(   'Song', id ) )

        pbar.set_postfix( action = 'downloading speech and song' )
        os.system( f'cd { dst } && wget { speech } && wget { song }' )

        pbar.set_postfix( action = 'unzipping speech' )
        with ZipFile( speech_path, 'r' ) as zf:
            zf.extractall( dst )

        if id != 18:
            pbar.set_postfix( action = 'unzipping song' )
            with ZipFile( song_path, 'r' ) as zf:
                zf.extractall( dst )

        os.remove( speech_path )
        if id != 18: os.remove( song_path )

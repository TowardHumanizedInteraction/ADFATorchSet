"""Metdata

The file contains methods to generate the dataset corresponding metadata.
"""
import pandas as pd
import os

from scipy.io import wavfile
from shutil import copyfile
from glob import glob
from tqdm import tqdm

"""Keys

Constants
---------
KEYS: List[ str ]
      List of the keys in to generate the metadata pandas dataframe.
"""
KEYS = [ 'id', 'sub_id', 'actor', 'modality', 'vocal', 'emotion', 'intensity',
         'statement', 'repetition', 'frame', 'face', 'audio', 'duration', 'sr',
         'fps' ]

def get_metadata( src: str, id: int ) -> dict:
    """Get Metadata

    Parameters
    ----------
    src: str
         Sequence path.
    id : int
         Sequence id.

    Returns
    -------
    dico: dict
          Dictionary containing the metadata for the given sequence.
    """
    seq        = src.split( '/' )[ -1 ]
    info       = seq.split( '-' )

    actor      = int( info[ 6 ] )
    modality   = int( info[ 0 ] )
    vocal      = int( info[ 1 ] )
    emotion    = int( info[ 2 ] )
    intensity  = int( info[ 3 ] )
    statement  = int( info[ 4 ] )
    repetition = int( info[ 5 ] )

    wav        = os.path.join( src, 'audio.wav' )
    pngs       = sorted( glob( os.path.join( src, '*.png' ) ) )
    npzs       = sorted( glob( os.path.join( src, '*.npz' ) ) )
    n          = min( len( pngs ), len( npzs ) )

    sr, x      = wavfile.read( wav )
    duration   = max( x.shape ) / sr

    dico       = { key: [ ] for key in KEYS }

    for i in range( n ):
        dico[ 'id'         ].append( id )
        dico[ 'sub_id'     ].append( i )
        dico[ 'actor'      ].append( actor )
        dico[ 'modality'   ].append( modality )
        dico[ 'vocal'      ].append( vocal )
        dico[ 'emotion'    ].append( emotion )
        dico[ 'intensity'  ].append( intensity )
        dico[ 'statement'  ].append( statement )
        dico[ 'repetition' ].append( repetition )
        dico[ 'frame'      ].append( pngs[ i ] )
        dico[ 'face'       ].append( npzs[ i ] )
        dico[ 'audio'      ].append( wav )
        dico[ 'duration'   ].append( duration )
        dico[ 'sr'         ].append( sr )
        dico[ 'fps'        ].append( 30 )

    return dico

def extract_metadata( src: str ):
    """Extract Metadata

    Parameters
    ----------
    src: str
         Path to the extracted dataset.
    """

    df     = pd.DataFrame( { key: [ ] for key in KEYS } )
    actors = [ os.path.join( src, actor ) for actor in os.listdir( src ) ]

    for actor in tqdm( sorted( actors ), desc = 'Generating metadata' ):
        sequences = [ os.path.join( actor, seq ) for seq in os.listdir( actor ) ]
        sequences = [ seq for seq in sequences if glob( os.path.join( seq, '*.npz' ) ) ]

        for i, sequence in enumerate( sorted( sequences ) ):
            dico = get_metadata( sequence, i )
            _df  = pd.DataFrame( dico )
            df   = pd.concat( [ df, _df ] )

    df[ 'frame' ] = df[ 'frame' ].apply( lambda x: os.path.relpath( x, src ) )
    df[ 'face' ]  = df[ 'face'  ].apply( lambda x: os.path.relpath( x, src ) )
    df[ 'audio' ] = df[ 'audio' ].apply( lambda x: os.path.relpath( x, src ) )

    df.to_csv( os.path.join( src, 'metadata.csv' ) )
    copyfile( './base.csv', os.path.join( src, 'base.csv' ) )

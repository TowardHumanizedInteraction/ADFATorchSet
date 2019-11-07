"""Preprocess

The file contains the preprocessing pipeline in order to transform the dataset
to the frontal facial plane.
"""
import pandas as pd
import numpy as np
import os

from multiprocessing import Pool
from typing import Tuple
from tqdm import tqdm

class Face:
    """Face

    Facial keypoints.

    Attributes
    ----------
    NOSE : List[ int ]
    REYE : int
    LEYE : int
    RTEAR: int
    LTEAR: int
    RBEAR: int
    LBEAR: int
    """
    NOSE  = [ 8201, 8202, 8203 ]

    REYE  = 14323
    LEYE  = 1956

    RTEAR = 34671
    LTEAR = 19893

    RBEAR = 34908
    LBEAR = 19720

def compute_X( face: Face, verts: np.ndarray ) -> np.ndarray:
    """Compute X axis

    Parameters
    ----------
    face : Face
           Facial Keypoints
    verts: np.ndarray
           Face vertices

    Returns
    -------
    X: np.ndarray
       Computed X axis
    """
    X1 = verts[ :, face.RTEAR ] - verts[ :, face.LTEAR ]
    X1 = X1 / np.sqrt( ( X1 ** 2 ).sum( axis = 0 ) )

    X2 = verts[ :, face.RBEAR ] - verts[ :, face.LBEAR ]
    X2 = X2 / np.sqrt( ( X2 ** 2 ).sum( axis = 0 ) )

    X  = ( X1 + X2 ) / 2

    return X

def compute_Z( face: Face, verts: np.ndarray ) -> np.ndarray:
    """Compute Z axis

    Parameters
    ----------
    face : Face
           Facial Keypoints
    verts: np.ndarray
           Face vertices

    Returns
    -------
    Z: np.ndarray
       Computed Z axis
    """
    M = ( verts[ :, face.RBEAR ] + verts[ :, face.LBEAR ] ) * .5

    Z = verts[ :, face.NOSE ].mean( axis = 1 ) - M
    Z = Z / np.sqrt( ( Z ** 2 ).sum( axis = 0 ) )

    return Z

def compute_Y( X: np.ndarray, Z: np.ndarray ) -> np.ndarray:
    """Compute Y axis

    Parameters
    ----------
    X: np.ndarray
       Computed X axis
    Z: np.ndarray
       Computed Z axis

    Returns
    -------
    Y: np.ndarray
       Computed Y axis
    """
    Y = np.cross( Z, X )
    Y = Y / np.sqrt( ( Y ** 2 ).sum( axis = 0 ) )

    return Y

def project( verts: np.ndarray ) -> np.ndarray:
    """Compute X axis

    Parameters
    ----------
    verts: np.ndarray
           Face vertices

    Returns
    -------
    verts: np.ndarray
           Facial vertices projected to the frontal facial plane.
    """
    W = np.array( [ 0, 0, 0, 1 ] )

    X = compute_X( verts )
    Z = compute_Z( verts )
    Y = compute_Y( X, Z )

    X = np.append( X, [ 0 ] )
    Y = np.append( Y, [ 0 ] )
    Z = np.append( Z, [ 0 ] )

    T = np.array( [ X, Y, Z, W ] ).T

    return np.array( [
        np.append( p, [ 1 ] ) @ T
        for p in verts.T
    ] )[ :, :3 ].T

def offset( face: Face, verts: np.ndarray ) -> np.ndarray:
    """Offset to center

    Parameters
    ----------
    face : Face
           Facial Keypoints
    verts: np.ndarray
           Face vertices

    Returns
    -------
    verts: np.ndarray
           Facial vertices centered to the eyes.
    """
    C = .5 * ( verts[ :, face.REYE ] + verts[ :, face.LEYE ] )

    return ( verts.T - C ).T

def rescale( face: Face, p_verts: np.ndarray, n_verts: np.ndarray ) -> np.ndarray:
    """Scale stabilization

    Parameters
    ----------
    face   : Face
             Facial Keypoints
    p_verts: np.ndarray
             Previous Face vertices
    n_verts: np.ndarray
             Current Face vertices

    Returns
    -------
    verts: np.ndarray
           Facial vertices with stabilized scale.
    """
    p = np.abs( p_verts[ 0, face.RBEAR ] - p_verts[ 0, face.LBEAR ] )
    n = np.abs( n_verts[ 0, face.RBEAR ] - n_verts[ 0, face.LBEAR ] )

    return ( n_verts.T * p / n ).T

def preprocess_one( verts: np.ndarray, p_verts: np.ndarray ) -> np.ndarray:
    """Preprocess one Frame

    Parameters
    ----------
    verts  : np.ndarray
             Current Face vertices
    p_verts: np.ndarray
             Previous Face vertices

    Returns
    -------
    verts: np.ndarray
           Facial vertices transformed.
    """
    verts = project( verts )
    verts = offset( verts )

    if p_verts is not None:
        verts = rescale( p_verts, verts )

    return verts.astype( np.float32 )

def preprocess_seq( params: Tuple ) -> None:
    """Preprocess one Sequence

    Parameters
    ----------
    params: Tuple
            metadata, m_actor, sr, actor, id, B from preprocess_3ddfa fun
    """
    metadata   = params[ 0 ]
    m_actor    = params[ 1 ]
    src        = params[ 2 ]
    actor      = params[ 3 ]
    id         = params[ 4 ]
    B          = params[ 5 ]

    m_actor_id = m_actor[ m_actor.id == id ]
    p_verts    = B

    for sub_id in sorted( m_actor_id.sub_id.unique( ) ):
        data    = metadata[ ( metadata.actor == actor ) & ( metadata.id == id ) & ( metadata.sub_id == sub_id ) ]
        path    = os.path.join( src, data.face.values[ 0 ] )

        verts   = np.load( path )[ 'vertices' ]
        tri     = np.load( path )[ 'triangles' ]

        n_verts = preprocess_one( verts, p_verts )
        p_verts = n_verts

        np.savez( path, vertices = n_verts, triangles = tri  )

def preprocess_3ddfa( src: str ) -> None:
    """Call

    Parameters
    ----------
    src: str
         Source path to the extracted folder
    """
    base     = pd.read_csv( os.path.join( src, 'base.csv' ), sep = ';' )
    metadata = pd.read_csv( os.path.join( src, 'metadata.csv' ), index_col = 0 )
    actors   = metadata.actor.unique( )

    for actor in tqdm( actors, desc = 'Preprocessing Actor' ):
        m_actor = metadata[ metadata.actor == actor ]
        B       = os.path.join( src, base[ base.actor == int( actor ) ].base.values[ 0 ] )
        B       = preprocess_one( np.load( B )[ 'vertices' ], None )

        with Pool( ) as pool:
            ids     = sorted( m_actor.id.unique( ) )
            iter    = [
                ( metadata, m_actor, src, actor, id, B )
                for id in ids
            ]
            n       = len( ids )
            imap    = pool.imap_unordered( preprocess_seq, iter )
            pbar    = tqdm( imap, total = n, desc = 'Preprocessing Sequence' )
            _       = list( pbar )

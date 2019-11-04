"""Preprocess

The file contains the preprocessing pipeline in order to transform the dataset
to the frontal facial plane.
"""
import pandas as pd
import numpy as np
import os

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

class Preprocess:
    """Preprocess

    Attributes
    ----------
    FACE: Face
          Facial Keypoints
    """

    def __init__( self: 'Preprocess' ) -> None:
        """Init
        """
        self.FACE = Face

    def compute_X( self: 'Preprocess', verts: np.ndarray ) -> np.ndarray:
        """Compute X axis

        Parameters
        ----------
        verts: np.ndarray
               Face vertices

        Returns
        -------
        X: np.ndarray
           Computed X axis
        """
        X1 = verts[ :, self.FACE.RTEAR ] - verts[ :, self.FACE.LTEAR ]
        X1 = X1 / np.sqrt( ( X1 ** 2 ).sum( axis = 0 ) )

        X2 = verts[ :, self.FACE.RBEAR ] - verts[ :, self.FACE.LBEAR ]
        X2 = X2 / np.sqrt( ( X2 ** 2 ).sum( axis = 0 ) )

        X  = ( X1 + X2 ) / 2

        return X

    def compute_Z( self: 'Preprocess', verts: np.ndarray ) -> np.ndarray:
        """Compute Z axis

        Parameters
        ----------
        verts: np.ndarray
               Face vertices

        Returns
        -------
        Z: np.ndarray
           Computed Z axis
        """
        M = ( verts[ :, self.FACE.RBEAR ] + verts[ :, self.FACE.LBEAR ] ) * .5

        Z = verts[ :, self.FACE.NOSE ].mean( axis = 1 ) - M
        Z = Z / np.sqrt( ( Z ** 2 ).sum( axis = 0 ) )

        return Z

    def compute_Y( self: 'Preprocess', X: np.ndarray, Z: np.ndarray ) -> np.ndarray:
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

    def project( self: 'Preprocess', verts: np.ndarray ) -> np.ndarray:
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

        X = self.compute_X( verts )
        Z = self.compute_Z( verts )
        Y = self.compute_Y( X, Z )

        X = np.append( X, [ 0 ] )
        Y = np.append( Y, [ 0 ] )
        Z = np.append( Z, [ 0 ] )

        T = np.array( [ X, Y, Z, W ] ).T

        return np.array( [
            np.append( p, [ 1 ] ) @ T
            for p in verts.T
        ] )[ :, :3 ].T

    def offset( self: 'Preprocess', verts: np.ndarray ) -> np.ndarray:
        """Offset to center

        Parameters
        ----------
        verts: np.ndarray
               Face vertices

        Returns
        -------
        verts: np.ndarray
               Facial vertices centered to the eyes.
        """
        C = .5 * ( verts[ :, self.FACE.REYE ] + verts[ :, self.FACE.LEYE ] )

        return ( verts.T - C ).T

    def rescale( self: 'Preprocess', p_verts: np.ndarray, n_verts: np.ndarray ) -> np.ndarray:
        """Scale stabilization

        Parameters
        ----------
        p_verts: np.ndarray
                 Previous Face vertices
        n_verts: np.ndarray
                 Current Face vertices

        Returns
        -------
        verts: np.ndarray
               Facial vertices with stabilized scale.
        """
        p = np.abs( p_verts[ 0, self.FACE.RBEAR ] - p_verts[ 0, self.FACE.LBEAR ] )
        n = np.abs( n_verts[ 0, self.FACE.RBEAR ] - n_verts[ 0, self.FACE.LBEAR ] )

        return ( n_verts.T * p / n ).T

    def preprocess( self: 'Preprocess', verts: np.ndarray, p_verts: np.ndarray ) -> np.ndarray:
        """Scale stabilization

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
        verts = self.project( verts )
        verts = self.offset( verts )

        if p_verts is not None:
            verts = self.rescale( p_verts, verts )

        return verts.astype( np.float32 )

    def __call__( self: 'Preprocess', src: str ) -> None:
        """Call

        Parameters
        ----------
        src: str
             Source path to the extracted folder
        """
        base     = pd.read_csv( os.path.join( src, 'base.csv' ), sep = ';' )
        metadata = pd.read_csv( os.path.join( src, 'metadata.csv' ), index_col = 0 )
        actors   = metadata.actor.unique( )

        for actor in tqdm( actors, desc = 'Preprocessing' ):
            m_actor = metadata[ metadata.actor == actor ]
            base    = os.path.join( src, base[ base.actor == int( actor ) ].base.values[ 0 ] )
            B       = self.preprocess( np.load( base )[ 'vertices' ], None )

            for id in sorted( m_actor.id.unique( ) ):
                m_actor_id = m_actor[ m_actor.id == id ]
                p_verts    = B

                for sub_id in sorted( m_actor_id.sub_id.unique( ) ):
                    data    = metadata[ ( metadata.actor == actor ) & ( metadata.id == id ) & ( metadata.sub_id == sub_id ) ]
                    path    = os.path.join( src, data.face.values[ 0 ] )

                    verts   = np.load( path )[ 'vertices' ]
                    tri     = np.load( path )[ 'triangles' ]

                    n_verts = self.preprocess( verts, p_verts )
                    p_verts = n_verts

                    np.savez( path, vertices = n_verts, triangles = tri  )

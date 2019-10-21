"""3DDFA

The file provides methods to generate 3D faces vertices out of monocular
images using the 3DDFA project [1] for the RAVDESS dataset.

[1] https://github.com/cleardusk/3DDFA
"""
import sys
sys.path.append( './3DDFA' )

import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import scipy.io as sio
import mobilenet_v1
import numpy as np
import torch
import dlib
import cv2
import os

from utils.inference import parse_roi_box_from_landmark
from utils.inference import predict_dense
from utils.inference import crop_img
from utils.ddfa import NormalizeGjz
from utils.ddfa import ToTensorGjz
from multiprocessing import Pool
from typing import Tuple
from typing import List
from typing import Any
from tqdm import tqdm
from glob import glob

"""Paths

Constants
---------
GIT_3DDFA  : str
             Url to the 3DDFA github repository
URL_3DDFA  : str
             Url to the 68 face landmarks dlib predictor.
URL_RAVDESS: str
             Base url for the RAVDESS dataset zip files.
"""
MODEL_CHECKPOINT = './3DDFA/models/phase1_wpdc_vdc.pth.tar'
MODEL_LANDMARKS  = './3DDFA/models/shape_predictor_68_face_landmarks.dat'
ARCHITECTURE     = 'mobilenet_1'
TRI              = './3DDFA/visualize/tri.mat'

def block_print( ) -> None:
    """Block Print
    """
    sys.stdout = open( os.devnull, 'w' )

def enable_print( ) -> None:
    """Enable Print
    """
    sys.stdout = sys.__stdout__

def get_model( cuda: bool = True ) -> torch.nn.Module:
    """Get Model

    Parameters
    ----------
    cuda: bool
          Enable GPU for torch module.

    Returns
    -------
    model: torch.nn.Module
           Facial Reconstruction model from 3DDFA.
    """
    checkpoint      = torch.load(
        MODEL_CHECKPOINT,
        map_location = lambda storage, loc: storage
    )[ 'state_dict' ]
    model           = getattr( mobilenet_v1, ARCHITECTURE )( num_classes = 62 )
    model_dict      = model.state_dict()

    for k in checkpoint.keys( ):
        model_dict[ k.replace( 'module.', '' ) ] = checkpoint[ k ]
    model.load_state_dict( model_dict )

    cudnn.benchmark = cuda
    model           = model.cuda() if cuda else model
    model.eval()

    return model

def get_face_model( ) -> Tuple[ Any ]:
    """Get Face Model

    Returns
    -------
    face_regressor: dlib::object_detector
                    Dlib model for 68 Facial Landmark regressor.
    face_detector : dlib::object_detector
                    Dlib model for Facial detection.
    """
    face_regressor      = dlib.shape_predictor( MODEL_LANDMARKS )
    face_detector       = dlib.get_frontal_face_detector( )

    return face_regressor, face_detector

def get_inputs( files: List[ str ] ) -> Tuple[ torch.Tensor, List[ np.ndarray ] ]:
    """Get Inputs

    Parameters
    ----------
    files: List[ str ]
           List of frames to process.

    Returns
    -------
    inputs  : torch.Tensor
              Frame cropped to the face.
    imgs_roi: List[ np.ndarray ]
              Frame face rectangles ( region of interest ).
    """
    face_reg, face_det = get_face_model( )
    transform          = transforms.Compose( [
        ToTensorGjz( ),
        NormalizeGjz( mean = 127.5, std = 128 )
    ] )

    imgs_ori           = [ cv2.imread( file ) for file in files ]
    imgs_rect          = [ face_det( img )[ 0 ] for img in imgs_ori ]
    imgs_pts           = [
        face_reg( imgs_ori[ i ], imgs_rect[ i ] ).parts( )
        for i in range( len( imgs_ori ) )
    ]
    imgs_pts           = [
        np.array( [ [ pt.x, pt.y ] for pt in imgs_pts[ i ] ] ).T
        for i in range( len( imgs_ori ) )
    ]
    imgs_roi           = [
        parse_roi_box_from_landmark( pts )
        for pts in imgs_pts
    ]
    imgs_cropped       = [
        crop_img( imgs_ori[ i ], imgs_roi[ i ] )
        for i in range( len( imgs_ori ) )
    ]
    imgs_new           = [
        cv2.resize( img, dsize = ( 120, 120 ), interpolation = cv2.INTER_LINEAR )
        for img in imgs_cropped
    ]

    inputs             = [ transform( img ).unsqueeze( 0 ) for img in imgs_new ]
    inputs             = torch.cat( inputs, axis = 0 )

    return inputs, imgs_roi

def get_params( files: List[ str ], batch_size: int = 16, cuda: bool = True ) -> Tuple[ np.ndarray, List[ np.ndarray ] ]:
    """Get Params

    Parameters
    ----------
    files     : List[ str ]
                List of frames to process.
    batch_size: int
                default 16
                Batch Size for facial reconstruction inference.
    cuda      : bool
                Enable GPU for faster inference.

    Returns
    -------
    params  : np.ndarray
              Infered facial vertices for each frame.
    imgs_roi: List[ np.ndarray ]
              Frame face rectangles ( region of interest ).
    """
    model            = get_model( cuda )
    inputs, imgs_roi = get_inputs( files )

    with torch.no_grad():
        params = [ ]

        for s in range( 0, inputs.size( 0 ), batch_size ):
            e       = min( s + batch_size, inputs.size( 0 ) )
            _inputs = inputs[ s:e, ... ].cuda( ) if cuda else inputs[ s:e, ... ]
            _params = model( _inputs ).detach( ).cpu( ).numpy( ).astype( np.float32 )
            _params = _params.reshape( ( _params.shape[ 0 ], -1 ) )
            params.append( _params )

        params = np.concatenate( params, axis = 0 )

    return params, imgs_roi

def save_face( params: Tuple[ str, str, str, str, np.ndarray, np.ndarray ] ) -> None:
    """Save Face

    Parameters
    ----------
    params: Tuple[ str, str, str, str, np.ndarray, np.ndarray ]
            Source, Actor, Sequence, File, Vertices, Triangles
            Params to save Vertices and Triangles in npz format.
    """
    src, actor, seq, file, verts, tri = params
    name                              = file.split( '/' )[ -1 ].split( '.' )[ 0 ]
    path                              = os.path.join( src, actor )
    path                              = os.path.join( path, seq )
    path                              = os.path.join( path, f'{ name }.npz' )

    np.savez( path, vertices = verts, triangles = tri )

def get_incomplete( src: str, actor: str, sequence: str, files: List[ str ] ) -> None:
    """Get Incomplete

    Parameters
    ----------
    src     : str
    actor   : str
    sequence: str
    files   : List[ str ]
              List of videos to be extracted.

    Returns
    -------
    res: List[ str ]
         List of videos which have not been extracted yet.
    """
    base = os.path.join( os.path.join( src, actor ), sequence )
    res  = [
        file
        for file in files if \
        not os.path.isfile( os.path.join(
            base,
            f'{ file.split( "/" )[ -1 ].split( "." )[ 0 ] }.npz'
        ) )
    ]

    return res

def extract_3ddfa( src: str, batch_size: int = 16, cuda: bool = True ) -> None:
    """Extract 3DDFA

    Parameters
    ----------
    src       : str
                Where to find the dataset.
    batch_size: int
                default 16
                Batch Size for facial reconstruction inference.
    cuda      : bool
                Enable GPU for faster inference.

    """
    actors = os.listdir( src )

    a_pbar = tqdm( actors, desc = 'Actor' )
    for actor in a_pbar:
        sequences = os.listdir( os.path.join( src, actor ) )

        for sequence in sorted( sequences ):
            a_pbar.set_postfix( name = actor, seq = sequence )

            path             = os.path.join( os.path.join( src, actor ), sequence )
            files            = sorted( glob( os.path.join( path, '*.png' ) ) )
            files            = get_incomplete( src, actor, sequence, files )

            if not files:
                continue

            try:
                params, imgs_roi = get_params( sorted( files ), batch_size, cuda )
                tri              = sio.loadmat( TRI )[ 'tri' ]
                imgs_verts       = [
                    predict_dense( params[ i ], imgs_roi[ i ] )
                    for i in range( len( params ) )
                ]
            except:
                continue


            with Pool( ) as pool:
                n                = len( files )
                iter             = [
                    ( src, actor, sequence, files[ i ], imgs_verts[ i ], tri )
                    for i in range( len( files ) )
                ]
                imap             = pool.imap_unordered( save_face, iter )
                pbar             = tqdm( imap, total = n, desc = 'Saving Files' )
                _                = list( pbar )

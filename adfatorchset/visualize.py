# import numpy as np
# import os
#
# if __name__ == '__main__':
#     import sys
#     sys.path.append( './3DDFA' )
#
#     from utils.lighting import RenderPipeline
#     from tqdm import tqdm
#     from PIL import Image
#     from glob import glob
#
#     rend   = RenderPipeline( )
#     bckg   = np.zeros( ( 720, 1280, 3 ) )
#     actors = sorted( glob( './ravdess/faces/*' ) )
#
#     for actor in tqdm( actors, desc = 'Actors' ):
#         actor_name = actor.split( '/' )[ -1 ]
#         sequences  = sorted( glob( os.path.join( actor, '*' ) ) )
#
#         if not os.path.isdir( os.path.join( './test', actor_name ) ):
#             os.mkdir( os.path.join( './test', actor_name ) )
#
#         for sequence in tqdm( sequences, desc = 'Sequence' ):
#             sequence_name = sequence.split( '/' )[ -1 ]
#             apath         = os.path.join( os.path.join( os.path.join( './ravdess/audios', actor_name ), sequence_name ), 'audio.wav' )
#             fname         = os.path.join( os.path.join( './test', actor_name ), sequence_name )
#
#             if os.path.isfile( f'{ fname }.mp4' ):
#                 continue
#
#             faces         = [ os.path.join( sequence, f ) for f in sorted( os.listdir( sequence ) ) if '.npy' in f ]
#             data          = [ np.load( f, allow_pickle = True ).item( ) for f in tqdm( faces, desc = 'Reading Files' ) ]
#
#             for i in tqdm( range( len( data ) ), desc = 'Generating visuals' ):
#                 Image.fromarray( rend(
#                     np.ascontiguousarray( data[ i ][ 'vertices' ].T ),
#                     np.ascontiguousarray( data[ i ][ 'triangles' ].T - 1 )\
#                         .copy( order = 'C' )\
#                         .astype( np.int32 ),
#                     bckg
#                 ) ).save( f'./test/frame_{i:06d}.png' )
#
#             os.system( f'ffmpeg -framerate 30 -pattern_type glob -i "./test/*.png" -c:v libx264 -pix_fmt yuv420p ./test/tmp.mp4' )
#             os.system( f'ffmpeg -i ./test/tmp.mp4 -i { apath } -vcodec copy { fname }.mp4' )
#             os.system( 'rm ./test/*.png ./test/tmp.mp4' )

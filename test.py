# python setup.py build_ext --inplace
import testing
import cv2, numpy
from mve.core import Scene, View, Image
import mve.core
#s = Scene('tmp/b-daman/scene')
s = Scene()
s.load('tmp/b-daman/scene')
print(s)
#s.load(0)

views = s.views

#img = Image(640, 480, 3, mve.core.IMAGE_TYPE_UINT8)
#print(img)

img = numpy.ones((256,256,3), dtype=numpy.uint8) * 128

views[2].set_image("undist-L1", img)
views[10].remove_image("undist-L1")

for view in views:
    print(view)
    #print(view.id)
    cam = view.camera
    #if not view.valid:
    #    print("View[{}] has invalid camera".format(view.id))
    #print('View[{}]: CamPos = {}'.format(view.id, cam.position))
    #print(cam.translation)
    #print(cam.view_dir)
    #print(cam.world_to_cam_matrix)
    #print(cam.cam_to_world_matrix)
    #print(cam.world_to_cam_rotation)
    #print(cam.cam_to_world_rotation)
    #print(cam.focal_length)
    #print(cam.principal_point)
    #print(cam.pixel_aspect)
    #print(cam.distortion)
    #print(cam.get_calibration(width=1, height=1))
    img = view.get_image('undist-L1')
    #print(img)
    if img is not None:
        cv2.imshow("show", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
    else:
        print("{} has no image".format(view))
    view.cleanup_cache()

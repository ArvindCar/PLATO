import robomail.vision as vis
from frankapy import FrankaArm
import cv2
import PIL
from PIL import Image

if __name__ == '__main__':
    fa = FrankaArm()
    cam1 = vis.CameraClass(1, W=1280, H=720)
    transform_1w = cam1.get_cam_extrinsics()
    cimage1, _, _, _, _ = cam1.get_next_frame()
    cv2.imwrite('spoon_obj.png', cimage1)
    image = Image.open('spoon_obj.png')
    width, height = image.size
    new_width = 394
    new_height = 221
    re_image = image.resize((new_width, new_height))
    re_image.save('spoon_obj_re.png')

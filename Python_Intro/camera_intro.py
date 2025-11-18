from picamzero import Camera
import time

camera = Camera()
#camera.still_size = (1536, 864)
#camera.video_size = (1536, 864)
camera.flip_camera(hflip=True, vflip=True)
time.sleep(2)

#camera.take_photo("/home/steve/Desktop/pi_photos/photo.jpg")
print("starting video")
camera.record_video("/home/steve/Desktop/pi_photos/video", 7)
print("Done")
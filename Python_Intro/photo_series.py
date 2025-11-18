from picamzero import Camera
import time
import os

#Initialize the camera and set the desired destination folder for pictures
folder_name = os.fspath("/home/steve/Desktop/pi_photos/activity_10")
camera = Camera()
camera.flip_camera(hflip=True, vflip=True)
time.sleep(2)

try:
    #If the desired destination folder doesn't exist, create it
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    print("Taking Photos")
    i = 1
    while True:
        #Take a photo every 5 seconds and save it with a unique name
        filename = folder_name + "/image" + str(i) + ".jpg"
        camera.take_photo(filename)
        print("Photo " + str(i) + " taken")
        i += 1
        time.sleep(5)

except KeyboardInterrupt:
    print("Done")
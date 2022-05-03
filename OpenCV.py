from PIL import Image
from PIL import ImageGrab

size = (300, 300, 400, 400)
img = ImageGrab.grab(size)
img.save("cut.jpg")
img.show()
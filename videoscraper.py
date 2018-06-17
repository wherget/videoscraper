#!python3
#
# usage: videoscraper.py video-file fps logfile
#
# on debian(9)
#
# apt-get install virtualenv python3-virtualenv ffmpeg tesseract-ocr tesseract-ocr-eng libtesseract-dev
# virtualenv --python=python3 videoscraper
# source videoscraper/bin/activate
# pip install tesserpy image2pipe opencv-python streamlink
# streamlink -v -p python3 -a "videoscraper.py {filename} 5 output.log" -n twitch.tv/loadingreadyrun best
#

from multiprocessing import Queue
import sys
import image2pipe
import tesserpy
import cv2
from datetime import datetime
from PIL import Image

crops = {
  'speed':       dict(coords=[76, 39, 100,180], chars="0123456789",   invert=True, threshold=200),
  'lastlap':     dict(coords=[45, 26,  49,351], chars="0123456789",   invert=True, threshold=150),
  'lastlaptime': dict(coords=[90, 26, 162,351], chars="0123456789.:", invert=True, threshold=150),
  'position':    dict(coords=[65, 26,  69,612], chars="0123456789/",  invert=True, threshold=128),
  'laptime':     dict(coords=[142,47,  11,273], chars="0123456789:.", invert=True, threshold=150),
  'storm':       dict(coords=[64, 32,1595,23 ], chars="0123456789",   invert=False, threshold=100),
  'fuel':        dict(coords=[37, 20,  66,146], chars="0123456789.",  invert=True, threshold=180)
}

# debug dump images
dump = False

def process(image, invert=False, threshold=100):
	temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	if (threshold > 127):
		mode = cv2.THRESH_BINARY # high-contrast
	else:
		mode = cv2.THRESH_TOZERO # low-contrast
	ret, temp = cv2.threshold(temp, thresh=threshold, maxval=255, type=mode)
	if (invert):
		temp = cv2.bitwise_not(temp)
	return temp

def crop(image, coordinates):
	x_start = coordinates[2]
	x_end = x_start + coordinates[0]
	y_start = coordinates[3]
	y_end = y_start + coordinates[1]
	return image[y_start:y_end, x_start:x_end]

q = Queue()
decoder = image2pipe.images_from_url(q, sys.argv[1], fps=sys.argv[2], scale=(1920, 1080))
decoder.start()

with open(sys.argv[3], "a") as out:
	sys.stdout = out
	for frame_number, image in image2pipe.utils.yield_from_queue(q):
		tesseract = tesserpy.Tesseract("/usr/share/tesseract-ocr/tessdata",language="eng", oem=tesserpy.OEM_TESSERACT_ONLY)
		tesseract.tessedit_pageseg_mode = tesserpy.PSM_SINGLE_LINE
		frametime = datetime.utcnow().isoformat()
		print(frametime+" #"+str(frame_number), end=' ')
		for datum, settings in crops.items():
			dir(settings)
			cropped = crop(image, settings['coords'])
			image_for_datum = process(cropped, settings['invert'], settings['threshold'])
			if dump:
				pilimage = Image.fromarray(image_for_datum)
				pilimage.save(datum+"_"+str(frame_number)+".png")
			tesseract.set_image(image_for_datum)
			tesseract.tessedit_char_whitelist = settings['chars']
			recognized = tesseract.get_utf8_text()
			tesseract.clear()
			print("%s=(%s)" % (datum, recognized.strip()), end=' ')
		print(flush=True)
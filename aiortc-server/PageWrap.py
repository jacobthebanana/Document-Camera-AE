import numpy as np
import cv2
import imutils

sr = cv2.dnn_superres.DnnSuperResImpl_create()
# path = "EDSR_x3.pb"
path = "ESPCN_x3.pb"
# path = "ESPCN_x2.pb"

sr.readModel(path)
sr.setModel("espcn", 3)
# sr.setModel("espcn", 2)
# sr.setModel("edsr", 3)

from skimage.filters import threshold_local

# Copied verbatim from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	
    # now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
	return warped


# With reference to https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def processFrame(frame):
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    ratio = frame.shape[0] / 500.0
    orig = frame.copy()
    image = imutils.resize(frame, height=500)

    # Cnvert the image to grayscale, apply blur, and find edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # edged = cv2.Canny(gray, 75, 200)
    edged = cv2.Canny(image, 100, 200)

    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    polygons = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        if len(approx) == 2 * 2:
            if cv2.contourArea(contour) >= 0.3 * image.shape[0] * image.shape[1]:
                # print("Rectangle found!")
                sheetCorners = approx
                # cv2.imwrite("/tmp/output1.jpg", cv2.drawContours(image, [approx], -1, (0, 255, 0), 2))
                warped = four_point_transform(orig, sheetCorners.reshape(4, 2) * ratio)

                # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                # T = threshold_local(warped, 11, offset=10, method="gaussian")
                # warped = (warped > T).astype("uint8") * 255

                # warped = sr.upsample(warped)
				
                # cv2.imwrite("/dev/shm/output-7.jpg", warped)
                return cv2.drawContours(image, [approx], -1, (0, 255, 0), 2), warped, True

            # print(cv2.contourArea(contour), 0.3 * image.shape[0] * image.shape[1])
            return cv2.drawContours(image, [approx], -1, (0, 255, 255), 2), None, False
            # return image, False

        polygons.append(approx)
    
    # return None
    # return cv2.drawContours(image, polygons, -1, (0, 255, 0), 2)
    return image, None, False
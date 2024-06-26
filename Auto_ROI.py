
from PIL import Image
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import cv2
import numpy as np

image = Image.open('E:/RGB_Image.png')
print("Image size:",image.size)
image = np.array(image)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_green = np.array([25, 80, 80])
upper_green = np.array([100, 255, 200])
mask = cv2.inRange(hsv, lower_green, upper_green)
green_part = cv2.bitwise_and(image, image, mask=mask)
down_width = 1000 
down_height = 1000 
down_points = (down_width, down_height)
resized_down = cv2.resize(green_part, down_points, interpolation= cv2.INTER_LINEAR)
cv2.imshow('Image after removal of shaded and pale parts', resized_down) 
cv2.waitKey(0)

gray = cv2.cvtColor(green_part, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
mask = cv2.drawContours(
    np.zeros_like(gray), 
    [largest_contour],   
    contourIdx=-1,         
    color=255,             
    thickness=cv2.FILLED   
)
largest_contour_region = cv2.bitwise_and(image, image, mask=mask)
gray_region = cv2.cvtColor(largest_contour_region, cv2.COLOR_BGR2GRAY)
ret, triangle_threshold = cv2.threshold(gray_region,0,255,cv2.THRESH_BINARY, cv2.THRESH_TRIANGLE)
cv2.imshow('Largest contour region', triangle_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

M = cv2.moments(triangle_threshold)
centroid_x = int(M['m10'] / M['m00'])
centroid_y = int(M['m01'] / M['m00'])

roi_width = 10
roi_height =10
roi_x = max(50, centroid_x - roi_width // 2)
roi_y = max(50, centroid_y - roi_height //2)
roi_x2 = min(centroid_x + roi_width // 2, image.shape[1])
roi_y2 = min(centroid_y + roi_height // 2, image.shape[0])

roi_image = cv2.rectangle(image.copy(), (roi_x, roi_y), (roi_x2, roi_y2), (255, 0, 0), 1)
cv2.imshow('ROI', roi_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("ROI coordinates:", roi_x, roi_x2, roi_y,roi_y2)

##################################################################################################################

hyperspectral_image = envi.open("E:/Spectral_file.hdr")
print(hyperspectral_image)

band_index = 100
plt.imshow(hyperspectral_image[:, :, band_index])
plt.axis('on')
plt.title('Hyperspectral Image')
plt.colorbar()
plt.show()

roi_pixels = hyperspectral_image[roi_y:roi_y2, roi_x:roi_x2, :]
mean_spectrum = np.mean(roi_pixels, axis=(0, 1))

plt.plot(mean_spectrum)
plt.title('Mean Spectrum')
plt.xlabel('Band')
plt.ylabel('Intensity')
plt.show()

band_87 = hyperspectral_image[:, :, 86]
band_157 = hyperspectral_image[:, :, 156]
band_47 = hyperspectral_image[:, :, 46]
band_60 = hyperspectral_image[:, :, 59]
band_54 = hyperspectral_image[:, :, 53]
band_104 = hyperspectral_image[:, :, 103]
band_40 = hyperspectral_image[:, :, 39]

NDVI = (band_157 - band_87) / (band_157 + band_87)
PRI = (band_47-band_60)/(band_47+band_60)
ARI = 1/band_54-1/band_104
CRI = 1/band_40 - 1/band_54

plt.imshow(NDVI, cmap='jet')  
plt.colorbar(label='NDVI')
plt.title('Normalized Difference Vegetation Index (NDVI)')
plt.show()
plt.imshow(PRI, cmap='jet')  
plt.colorbar(label='PRI')
plt.title('Photochemical refeltance index (PRI)')
plt.show()
plt.imshow(ARI, cmap='jet')  
plt.colorbar(label='ARI')
plt.title('Anthocyanin reflectance index (ARI)')
plt.show()
plt.imshow(CRI, cmap='jet')  
plt.colorbar(label='CRI')
plt.title('Carotenoid reflectance index (ARI)')
plt.show()

band_87 = mean_spectrum[86]  
band_157 = mean_spectrum[156]  
band_47 = mean_spectrum[46]
band_60 = mean_spectrum[59]
band_54 = mean_spectrum[53]
band_104 = mean_spectrum[103]
band_40= mean_spectrum[39]

NDVI = (band_157 - band_87) / (band_157 + band_87)
PRI = (band_47-band_60)/(band_47+band_60)
ARI = 1/band_54-1/band_104
CRI = 1/band_40 - 1/band_54

print("NDVI value:", NDVI)
print("PRI value:", PRI)
print("ARI value:", ARI)
print('CRI value:', CRI)



import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import cv2

hdr_path = "E:/Data_test/2023-12-28_035--6-19/results/REFLECTANCE_2023-12-28_035.hdr" 
img = envi.open(hdr_path)
print(img)

hyperspectral_image = img.load()
band_index = 100

plt.imshow(hyperspectral_image[:, :, band_index], cmap='gray')
plt.axis('on')
plt.title('Hyperspectral Image (Band {})'.format(band_index))
plt.colorbar()
plt.show()

fromCenter = False
roi = cv2.selectROI("Select ROI", hyperspectral_image[:, :, band_index], fromCenter)
cv2.destroyAllWindows()
start_x, start_y, width, height = roi

print("ROI Coordinates:", start_x, start_y, width, height)
roi_pixels = hyperspectral_image[start_y:start_y+height, start_x:start_x+width, :]
mean_spectrum = np.mean(roi_pixels, axis=(0, 1))
plt.plot(mean_spectrum)
plt.title('Mean Spectrum from ROI')
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


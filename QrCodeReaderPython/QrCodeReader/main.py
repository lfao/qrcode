from __future__ import print_function
import cv2
import numpy
import os
import sys
from reedsolo import ReedSolomonError


from qr_detector import extract_matrix, QrDetectorError
from qr_decoder import extract_bit_array, extract_string, error_correction, get_version_size, get_format_info_data, QrDecoderError

# This tests all the images of the dataset in the directory
if __name__ == '__main__':
	success_counter = 0
	mypath = './Testset/'
	filepath_list_unfiltered = (os.path.join(mypath, f) for f in os.listdir(mypath))
	filepath_list = [f for f in filepath_list_unfiltered if os.path.isfile(f)]
	for i, filename in enumerate(filepath_list, 1):
		print(filename)
		image = cv2.imread(filename,-1)
		if image is False:
			print('could not open picture')
		else:
			try:
				bit_matrix = extract_matrix(image, 400)
				mask_index, ecc_level = get_format_info_data(bit_matrix)
				version, size = get_version_size(bit_matrix)

				raw_bit_array, (mask_matrix, dataarea_indicator, bit_matrix_unmasked) = extract_bit_array(bit_matrix, mask_index, True)

				cv2.imshow('mask', cv2.resize(numpy.logical_not(mask_matrix).astype(float), (size * 8, size * 8), interpolation = cv2.INTER_NEAREST))
				cv2.imshow('data area', cv2.resize(dataarea_indicator.astype(float), (size * 8, size * 8), interpolation = cv2.INTER_NEAREST))
				cv2.imshow('data', cv2.resize(numpy.logical_not(bit_matrix_unmasked).astype(float), (size * 8, size * 8), interpolation = cv2.INTER_NEAREST))
					
				bit_array = error_correction(raw_bit_array, version, ecc_level)
				string = extract_string(bit_array, version) 
			except QrDetectorError as e:
				print('Error while detecting occurred: {}'.format(e), file = sys.stderr)
			except ReedSolomonError as e:
				print('Error while applying error correction occurred: {}'.format(e), file = sys.stderr)
			except QrDecoderError as e:
				print('Error while decoding occurred: {}'.format(e), file = sys.stderr)
			else:
				success_counter += 1
				print(string)
			finally:
				print('{} of {} succeeded. Success rate: {}%'.format(success_counter, i, round(float(success_counter) / i * 100, 2)))  

		cv2.waitKey(0) # you have to click into a picture and press a button for viewing the next
					   # picture
		cv2.destroyAllWindows()

	cv2.waitKey(0)


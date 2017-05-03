import numpy
import cv2
import os

from qr_detector import extract_matrix
from qr_decoder import extract_bit_array, extract_string, error_correction, get_version_size, get_format_info_data

if __name__ == '__main__':
    good_counter = 0
    mypath = './Testset/'
    files = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    for i, filename in enumerate(files):
        print filename
        image = cv2.imread(filename,-1)
        if image is False:
            print "could not open picture"
        else:
            bit_matrix = extract_matrix(image, 400)
            if bit_matrix is not None:

                try:
                    mask_index, ecc_level = get_format_info_data(bit_matrix)
                    version, size = get_version_size(bit_matrix)

                    bit_array_raw = extract_bit_array(bit_matrix, mask_index, True)
                    bit_array = error_correction(bit_array_raw, version, ecc_level)
                    string = extract_string(bit_array, version)
                    good_counter += 1
                    print i + 1, good_counter, float(good_counter)/(i + 1), string
                    
                except Exception as e:
                    print e
        #cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.waitKey(0)


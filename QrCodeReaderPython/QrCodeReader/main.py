import numpy
import cv2
import itertools
import operator
import functools
import os

from qr_detector import extract_matrix
from qr_decoder import extract_bit_array, extract_string, error_correction, get_version_size, get_format_info_data

def camera_loop():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print "Unable to open camera"
    
    output = ""
    while True:
        retval, image = capture.read()
        try:
            bit_matrix = extract_matrix(image)
            if bit_matrix is not None:
                mask_index, ecc_level = get_format_info_data(bit_matrix)
                version, size = get_version_size(bit_matrix)

                bit_array_raw = extract_bit_array(bit_matrix, mask_index)
                bit_array = error_correction(bit_array_raw, version, ecc_level)
                string = extract_string(bit_array, version)
        
                if string != output:
                    output = string
                    print output
        except Exception as e:
            print e

if __name__ == '__main__':

    #camera_loop()
    #filename = '45degree.jpg' # easy
    filename = 'IMG_2713.jpg' # chair
    #filename = 'IMG_2717.JPG' # wall
    #filename = 'IMG_2716.JPG' # keyboard , little extrapolation error
    #filename = 'IMG_2712.JPG' # wall, not flat, very high slope , little warping error    
    #filename = "QR5_gedreht.png"
    #filename = "chart.png"
    #filename = "alphanumeric.png"
    filename = "IMG_2727.JPG"
    filename = "4.JPG"
    # fp error # 2 3 9 14
    #algo ohne # 1 5 7 8
    #algo mit  # 10 11
        

    #files = [file for file in (os.path.join(['./Testset', f]) for f in os.listdir('./Testset/')) if os.path.isfile(file)]
    good_counter = 0
    #print files
    #print os.listdir('./Testset/')
    mypath = './Testset/'
    files = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    #print files
    for i, filename in enumerate(files):
        print filename
        image = cv2.imread(filename,-1)
        if image is False:
            print "could not open picture"
        else:
            bit_matrix = extract_matrix(image, 400)

            if bit_matrix is not None and True:
                try:
                    mask_index, ecc_level = get_format_info_data(bit_matrix)
                    version, size = get_version_size(bit_matrix)

                    bit_array_raw = extract_bit_array(bit_matrix, mask_index)
                    bit_array = error_correction(bit_array_raw, version, ecc_level)
                    string = extract_string(bit_array, version)
                    good_counter += 1
                    print i + 1, good_counter, float(good_counter)/(i + 1), string
                    
                except Exception as e:
                    print e
        cv2.waitKey(0)
        cv2.destroyAllWindows()


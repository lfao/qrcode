import numpy
import cv2
import itertools
import operator
import functools

from extract_qr_matrix_from_image import extract_qr_matrix_from_image
from extract_qr_data_from_matrix import extract_bit_array, extract_string, error_correction, get_version_size, get_format_info_data

def camera_loop():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print "Unable to open camera"
    
    output = ""
    while True:
        retval, image = capture.read()
        binary = extract_qr_matrix(image)
        if binary is not None:
            int_list, mode_index = extract_string(*extract_stream(binary))
            if mode_index == 2:
                new_output = "".join(chr(item) for item in int_list)
            if new_output != output:
                print new_output
                output = new_output
        
    

if __name__ == '__main__':

    #filename = '45degree.jpg' # easy
    #filename = 'IMG_2713.jpg' # chair
    #filename = 'IMG_2717.JPG' # wall
    #filename = 'IMG_2716.JPG' # keyboard , little extrapolation error
    #filename = 'IMG_2712.JPG' # wall, not flat, very high slope , little warping error    
    filename = "QR5.png"
    #filename = "chart.png"
    #filename = "alphanumeric.png"
    #filename = "IMG_2728.JPG"


    image = cv2.imread(filename,-1)
    
    bit_matrix = extract_qr_matrix_from_image(image, 400)

    mask_index, ecc_level = get_format_info_data(bit_matrix)
    version, size = get_version_size(bit_matrix)

    bit_array_raw = extract_bit_array(bit_matrix, mask_index)
    bit_array = error_correction(bit_array_raw, version, ecc_level)
    string = extract_string(bit_array, version)

    print string

    cv2.waitKey(0)



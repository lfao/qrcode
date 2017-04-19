import numpy
import cv2
import itertools
import functools
import operator
import math

def extract_qr_bin(image, output = False):
    TL, TR, BL, BR = range(4)

    image_resized = cv2.resize(image, (800, 800))
    image_gray = cv2.cvtColor(image_resized,cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(image_gray,100,200)
    _, contours, [hierachy] = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # calculating the center of ech contour
    contour_centers = numpy.array([ (moments['m10']/moments['m00'], moments['m01']/moments['m00']) if moments['m00'] != 0 else (float('inf'), float('inf')) 
        for moments in  (cv2.moments(contour, False) for contour in contours) ])
    
    # finding Finder Pattern which is a 5 times nested contour
    def getDept(index):
        count = 0;
        while hierachy[index][2] >= 0:
            index = hierachy[index][2];
            count +=1
        return count
    
    marks = [i for i in xrange(len(hierachy)) if getDept(i) >= 5]
    if len(marks) != 3: # check if 3 and only 3 pattern have been found
        print("Detected {} of 3 required pattern".format(len(marks)))
        return None
    
    # matching the Finter Pattern to the corners
    distance_iterator = ((numpy.linalg.norm(contour_centers[bottomleft] - contour_centers[topright]), (topleft, topright, bottomleft)) 
                        for bottomleft, topleft, topright in itertools.permutations(marks)
                        if 0 < numpy.cross(contour_centers[topright] - contour_centers[topleft], contour_centers[bottomleft] - contour_centers[topleft]))

    _ , point_list = max(distance_iterator)

    # calculating horizontal and vertical vectors for the alligned qr code
    topleft, topright, bottomleft = point_list
    horizontal_vector = contour_centers[topright] - contour_centers[topleft]
    verticial_vector = contour_centers[bottomleft] - contour_centers[topleft]


    # checking if size is enough for getting good values
    if any(cv2.contourArea(contours[pattern]) < 10 for pattern in point_list):
        print("Some of the detected pattern are to small.")
        return None
    
    # extracting 4 corners for each pattern
    contour_center_tuple_list = ((contours[pattern], contour_centers[pattern]) for pattern in point_list)
    def pattern_iterable():
        for contour, center in contour_center_tuple_list:
            categorie_distance_tuple_list = (((numpy.cross(horizontal_vector, contour_point - center) > 0, numpy.cross(verticial_vector, contour_point - center) < 0), 
                                               numpy.linalg.norm(center - contour_point), contour_point) for [contour_point] in contour)
            corner_selection_tuple_list = itertools.groupby(sorted(categorie_distance_tuple_list, key = operator.itemgetter(0)), operator.itemgetter(0))
            corner_points_tuple_list = (max(values, key = operator.itemgetter(1)) for categorie, values in corner_selection_tuple_list)
            corner_coordinate_list = [coordinates for _ , _  , coordinates in corner_points_tuple_list]
            yield corner_coordinate_list
            
    pattern_corner_list = numpy.array(list(pattern_iterable()))     

    # extrapolation of the bottom right corner
    tr_r_dif = pattern_corner_list[TR][TR] - pattern_corner_list[TR][BR]
    bl_b_dif = pattern_corner_list[BL][BL] - pattern_corner_list[BL][BR]   
    t = float(numpy.cross(pattern_corner_list[BL][BL] - pattern_corner_list[TR][TR], bl_b_dif)) / numpy.cross(tr_r_dif, bl_b_dif)
    br_br = pattern_corner_list[TR][TR] + t * tr_r_dif

    # defining the warp source quadrangle
    source = numpy.array([pattern_corner_list[TL][TL], pattern_corner_list[TR][TR], br_br, pattern_corner_list[BL][BL]], dtype = "float32")
    
    # calculating the number of pixels in the clean qr code
    pattern_average = numpy.mean([numpy.linalg.norm(pattern_corner_list[i][j]-pattern_corner_list[i][k]) for i in xrange(3) for j, k in [(0,1),(2,3),(0,2),(1,3)]])
    size_average    = numpy.mean([numpy.linalg.norm(pattern_corner_list[TL][TL]-pattern_corner_list[TR][TR]),  
                                  numpy.linalg.norm(pattern_corner_list[TL][BL]-pattern_corner_list[TR][BR]), 
                                  numpy.linalg.norm(pattern_corner_list[TL][TL]-pattern_corner_list[BL][BL]), 
                                  numpy.linalg.norm(pattern_corner_list[TL][TR]-pattern_corner_list[BL][BR])])
    pixelcount = int(round(size_average / pattern_average * 7))

    
    # defining the warp destination square which is 8*8 times the number of pixels in the clean qr code
    temp_warp_size = pixelcount * 8
    destination = numpy.array([(0, 0), (temp_warp_size, 0), (temp_warp_size, temp_warp_size), (0, temp_warp_size)], dtype = "float32")
    
    # doing the warping and thresholding
    warp_matrix = cv2.getPerspectiveTransform(source, destination);
    bigqr_nottresholded = cv2.warpPerspective(image_gray, warp_matrix, (temp_warp_size, temp_warp_size));	
    _ ,bigqr = cv2.threshold(bigqr_nottresholded, 127, 255, cv2.THRESH_BINARY);
    
    #resizing to the real amount of pixels and thresholding again
    qr_notresholded = cv2.resize(bigqr, (pixelcount, pixelcount))
    a ,qr = cv2.threshold(qr_notresholded, 127, 255, cv2.THRESH_BINARY);
    
    # extracting the data
    data = numpy.asarray(qr, dtype="bool")

    if output:
        cv2.imshow("resized",image_resized)
        cv2.imshow("gray",image_gray )
        cv2.imshow("canny", edges)
        cv2.imshow("bigqr_nottresholded", bigqr_nottresholded)
        cv2.imshow("bigqr", bigqr)
        cv2.imshow("qr", cv2.resize(qr, (temp_warp_size, temp_warp_size), interpolation = cv2.INTER_NEAREST))
    return data

def extract_qr_content(binary):
    dimension , _ = binary.shape
    version_field = binary[dimension - 9 : dimension - 12 : -1, 5::-1]
    x = version_field.reshape((18,))
    print x
    print numpy.sum(2**numpy.arange(len(x))*x)


if __name__ == '__main__':
    #filename = 'sample.jpg'
    filename = 'IMG_2713.jpg'
    #filename = 'IMG_2713gedreht.JPG'
    #filename = '45degree.jpg'
    image = cv2.imread(filename,-1)
    
    #rows,cols, _ = image.shape
    #angle = 90
    #M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    #image = cv2.warpAffine(image,M,(cols,rows))

    binary = extract_qr_bin(image, True)
    if image is numpy.array:
        extract_qr_content(binary)
        # decoding
    #qr = qrtools.QR()
    #qr.cecode(qr_thres_small)
    #print qr.data

    #    return p + t*r
    cv2.waitKey(0)
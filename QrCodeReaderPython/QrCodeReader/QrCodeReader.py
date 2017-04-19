import numpy
import cv2
import itertools
import functools
import operator
import math

def extract_qr_bin(image, output = True):
    TL, TR, BL, BR = range(4)

    image2 = cv2.resize(image, (800, 800))
    gray = cv2.cvtColor(image2,cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,100,200)
    cv2.imshow("canny", edges)
    _, contours, [hierachy] = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contour_centers = numpy.array([ (moments['m10']/moments['m00'], moments['m01']/moments['m00']) if moments['m00'] != 0 else (float('inf'), float('inf')) 
        for moments in  (cv2.moments(contour, False) for contour in contours) ])
    
    def getDept(index):
        count = 0;
        while hierachy[index][2] >= 0:
            index = hierachy[index][2];
            count +=1
        return count
    
    # Finding Finder Pattern
    marks = [i for i in xrange(len(hierachy)) if getDept(i) >= 5]
    if len(marks) != 3:
        print("Detected {} of 3 required pattern".format(len(marks)))
        return None
    
    distance_iterator = ((numpy.linalg.norm(contour_centers[bottomleft] - contour_centers[topright]), (topleft, topright, bottomleft)) 
                        for bottomleft, topleft, topright in itertools.permutations(marks)
                        if 0 < numpy.cross(contour_centers[topright] - contour_centers[topleft], contour_centers[bottomleft] - contour_centers[topleft]))

    _ , point_list = max(distance_iterator)

    topleft, topright, bottomleft = point_list
    horizontal_vector = contour_centers[topright] - contour_centers[topleft]
    verticial_vector = contour_centers[bottomleft] - contour_centers[topleft]

    contour_center_tuple_list = [(contours[pattern], contour_centers[pattern]) for pattern in point_list]
        
    if any(cv2.contourArea(contour) < 10 for contour, _ in contour_center_tuple_list):
        print("Some of the detected pattern are to small.")
        return None
    
    def pattern_iterable():
        for contour, center in contour_center_tuple_list:
            categorie_distance_tuple_list = (((numpy.cross(horizontal_vector, contour_point - center) > 0, numpy.cross(verticial_vector, contour_point - center) < 0), 
                                               numpy.linalg.norm(center - contour_point), contour_point) for [contour_point] in contour)
            corner_selection_tuple_list = itertools.groupby(sorted(categorie_distance_tuple_list, key = operator.itemgetter(0)), operator.itemgetter(0))
            corner_points_tuple_list = (max(values, key = operator.itemgetter(1)) for categorie, values in corner_selection_tuple_list)
            corner_coordinate_list = [coordinates for _ , _  , coordinates in corner_points_tuple_list]
            yield corner_coordinate_list
            
    pattern_corner_list = numpy.array(list(pattern_iterable()))     

    tr_r_dif = pattern_corner_list[TR][TR] - pattern_corner_list[TR][BR]
    bl_b_dif = pattern_corner_list[BL][BL] - pattern_corner_list[BL][BR]   
    
    t = float(numpy.cross(pattern_corner_list[BL][BL] - pattern_corner_list[TR][TR], bl_b_dif)) / numpy.cross(tr_r_dif, bl_b_dif)
    br_br = pattern_corner_list[TR][TR] + t * tr_r_dif

    source = numpy.array([pattern_corner_list[TL][TL], pattern_corner_list[TR][TR], br_br, pattern_corner_list[BL][BL]], dtype = "float32")

    pattern_average = numpy.mean([numpy.linalg.norm(pattern_corner_list[i][j]-pattern_corner_list[i][k]) for i in xrange(3) for j, k in [(0,1),(2,3),(0,2),(1,3)]])
    size_average    = numpy.mean([numpy.linalg.norm(pattern_corner_list[TL][TL]-pattern_corner_list[TR][TR]),  
                                  numpy.linalg.norm(pattern_corner_list[TL][BL]-pattern_corner_list[TR][BR]), 
                                  numpy.linalg.norm(pattern_corner_list[TL][TL]-pattern_corner_list[BL][BL]), 
                                  numpy.linalg.norm(pattern_corner_list[TL][TR]-pattern_corner_list[BL][BR])])
    pixelcount = int(round(size_average / pattern_average * 7))

    temp_warp_size = pixelcount * 8
    destination = numpy.array([(0, 0), (temp_warp_size, 0), (temp_warp_size, temp_warp_size), (0, temp_warp_size)], dtype = "float32")
    
    warp_matrix = cv2.getPerspectiveTransform(source, destination);
    qr_raw = cv2.warpPerspective(gray, warp_matrix, (temp_warp_size, temp_warp_size));	
    _ ,qr_thres = cv2.threshold(qr_raw, 127, 255, cv2.THRESH_BINARY);
    qr_small = cv2.resize(qr_thres, (pixelcount, pixelcount))
    a ,qr_thres_small = cv2.threshold(qr_small, 127, 255, cv2.THRESH_BINARY);
    cv2.imshow("qrtres", qr_thres)
    cv2.imshow("big", cv2.resize(qr_thres_small, (temp_warp_size, temp_warp_size), interpolation = cv2.INTER_NEAREST))
                
    data = numpy.asarray(qr_thres_small, dtype="bool")
    return data

def extract_qr_content(binary):
    dimension , _ = binary.shape
    version_field = binary[dimension - 9 : dimension - 12 : -1, 5::-1]
    x = version_field.reshape((18,))
    print x
    print numpy.sum(2**numpy.arange(len(x))*x)


if __name__ == '__main__':
    #filename = 'sample.jpg'
    #filename = 'IMG_2713.jpg'
    #filename = 'IMG_2713gedreht.JPG'
    filename = '45degree.jpg'
    image = cv2.imread(filename,-1)
    
    rows,cols, _ = image.shape
    angle = 90
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    image = cv2.warpAffine(image,M,(cols,rows))

    binary = extract_qr_bin(image)
    if image is numpy.array:
        extract_qr_content(binary)
        # decoding
    #qr = qrtools.QR()
    #qr.cecode(qr_thres_small)
    #print qr.data

    #    return p + t*r
    cv2.waitKey(0)
import numpy
import cv2
import itertools
import functools
import operator
import math
#import sys, qrcode
#from qrtools import QR
#import pyqrcode
#import qrtools
#import zbarlight

def extract_qr_bin(image, output = True):
    TL = 1
    TR = 2
    BL = 0
    BR = 3

    def order_gen((left, right)):
        return (0 if left else 2) + (1 if left == right else 0)

    image2 = cv2.resize(image, (800, 800))
    gray = cv2.cvtColor(image2,cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,100,200)
    cv2.imshow("canny", edges)
    edges, contours, hierachy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    mc = numpy.array([ (moments['m10']/moments['m00'], moments['m01']/moments['m00']) if moments['m00'] != 0 else (float('inf'), float('inf')) 
        for moments in  (cv2.moments(contour, False) for contour in contours) ])
    
    def getDept(index):
        count = 0;
        while hierachy[0][index][2] >= 0:
            index = hierachy[0][index][2];
            count +=1
        return count
    
    marks = [i for i in xrange(len(hierachy[0])) if getDept(i) >= 5]
    
    #print marks
    if len(marks) >= 3:
        marks = marks[0:3]

        def cross_product(pointA, pointB, pointC):
            return ((pointC[0] - pointB[0]) * (pointA[1] - pointB[1])) - ((pointC[1] - pointB[1]) * (pointA[0] - pointB[0]))
        
        distance_iterator = ((numpy.linalg.norm(mc[bottomleft] - mc[topright]), (bottomleft, topleft, topright)) 
                            for bottomleft, topleft, topright in itertools.permutations(marks) if 0 < cross_product(mc[bottomleft], mc[topleft], mc[topright]))# numpy.cross(mc[[bottomleft, topleft, topright],0], mc[[bottomleft, topleft, topright],1]))
        _ , point_list = max(distance_iterator)
        topleft, topright, bottomleft = point_list

        orientation = 3 - order_gen(mc[topright] > mc[bottomleft])
        dx, dy = mc[bottomleft] - mc[topright]
        slope = dy/dy

        relevant = [(contours[pattern], mc[pattern]) for pattern in point_list]
        
        if all(cv2.contourArea(contour) > 10 for contour, _ in relevant):
            def corners_iterable():
                for contour, point in relevant:
                    x, y, w, h = cv2.boundingRect(contour)
                    if abs(slope) > 5:
                        a, b, c, d = [x,y], [x + w, y], [x + w, y + h], [x, y + h]
                        #a, b, c, d = [x, y + h / 2], [x + w / 2 ,y], [x + w, y + h / 2], [x + w / 2, y + h]
 
                        def line_distance(p1, p2, p3):
                            return numpy.linalg.norm(numpy.cross(p2-p1, p1-p3))/numpy.linalg.norm(p2-p1)
                        categorie_tuple_list = (((line_distance(c, a, contour_point[0]), line_distance(b, d, contour_point[0])), contour_point[0]) for contour_point in contour)
                    
                    else:
                        middle = numpy.array([ x + w / 2, y + h / 2])
                        categorie_tuple_list = ((contour_point[0] < middle, contour_point[0]) for contour_point in contour)

                    categorie_distance_tuple_list = ((order_gen(bool_tuple), numpy.linalg.norm(point - contour_point), contour_point) for bool_tuple, contour_point in categorie_tuple_list)
                    corner_selection_tuple_list = itertools.groupby(sorted(categorie_distance_tuple_list, key = operator.itemgetter(0)), operator.itemgetter(0))
                    corner_points_tuple_list = (max(values, key = operator.itemgetter(1)) for categorie, values in corner_selection_tuple_list)
                    corner_coordinate_list = [coordinates for _ , _  , coordinates in corner_points_tuple_list]
                    yield corner_coordinate_list
            
            ccl = numpy.array(list(corners_iterable()))
            ccl = numpy.roll(ccl,orientation,1)
            
            tr_t_dif =   ccl[TR][TR] - ccl[TR][BR]
            bl_b_dif = ccl[BL][BL] - ccl[BL][BR]   
            
            if numpy.linalg.det([tr_t_dif, bl_b_dif]) != 0:
                t = numpy.linalg.det([ccl[BL][BL] - ccl[TR][TR], bl_b_dif]) / numpy.linalg.det([tr_t_dif, bl_b_dif])
                bl_bl = ccl[TR][TR] + t * tr_t_dif

                source = numpy.array([ccl[TL][TL], ccl[TR][TR], bl_bl, ccl[BL][BL]], dtype = "float32")
                #print "source", source

                pattern_average = numpy.mean([numpy.linalg.norm(ccl[i][j-1]-ccl[i][j]) for i in xrange(3) for j in xrange(4)])
                size_average =  numpy.mean([numpy.linalg.norm(ccl[TL][TL]-ccl[TR][TR]),  numpy.linalg.norm(ccl[TL][BL]-ccl[TR][BR]), numpy.linalg.norm(ccl[TL][TL]-ccl[BL][BL]), numpy.linalg.norm(ccl[TL][TR]-ccl[BL][BR])])
                pixelcount = int(round(size_average / pattern_average * 7))

                temp_size = pixelcount * 8
                destination = numpy.array([(0, 0), (temp_size, 0), (temp_size, temp_size), (0, temp_size)], dtype = "float32")
    
                warp_matrix = cv2.getPerspectiveTransform(source, destination);
                qr_raw = cv2.warpPerspective(image2, warp_matrix, (temp_size, temp_size));	
                qr_gray = cv2.cvtColor(qr_raw,cv2.COLOR_RGB2GRAY);
                _ ,qr_thres = cv2.threshold(qr_gray, 127, 255, cv2.THRESH_BINARY);
                qr_small = cv2.resize(qr_thres, (pixelcount, pixelcount))
                a ,qr_thres_small = cv2.threshold(qr_small, 127, 255, cv2.THRESH_BINARY);
                cv2.imshow("qrtres", qr_thres)
                cv2.imshow("big", cv2.resize(qr_thres_small, (temp_size, temp_size), interpolation = cv2.INTER_NEAREST))
                
                data = numpy.asarray(qr_thres_small, dtype="bool")
                return data
    return None



if __name__ == '__main__':
    #filename = 'sample.jpg'
    filename = 'IMG_2713.jpg'
    #filename = 'IMG_2713gedreht.JPG'
    image = cv2.imread(filename,-1)
    
    rows,cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
    image = cv2.warpAffine(image,M,(cols,rows))

    binary = extract_qr_bin(image)
    print binary
        # decoding
    #qr = qrtools.QR()
    #qr.cecode(qr_thres_small)
    #print qr.data

    #    return p + t*r
    cv2.waitKey(0)
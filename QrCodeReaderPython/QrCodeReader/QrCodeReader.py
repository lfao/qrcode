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



#
def tupelsub(left, right):
    return itertools.starmap(operator.sub,itertools.izip(left,right))

def euklied_distance (left, right):
    return math.sqrt(sum((leftelement - rightelement) ** 2 for leftelement, rightelement in itertools.izip(left,right)))

def line_slope(M, L):
    if M[0] != L[0]:
        return (M[1] - L[1]) / (M[0] - L[0])
    else:
        return float("inf")

def line_distance(L, M, J):
    #print L, M, J
    a = -line_slope(L, M)
    b = 1.0
    c = line_slope(L, M) * L[0] - L[1]
    #print L, M, J,  a, b, c
    return (a * J[0] + (b * J[1]) + c) / math.sqrt((a * a) + (b * b))

def line_distance_sign(L, M, J):

    a = -line_slope(L, M)
    b = 1.0
    c = line_slope(L, M) * L[0] - L[1]
    #print L, M, J,  a, b, c
    return (a * J[0] + (b * J[1]) + c) > 0

def cross_product(pointA, pointB, pointC):
    #print pointA, pointB, pointC
    return ((pointC[0] - pointB[0]) * (pointA[1] - pointB[1])) - ((pointC[1] - pointB[1]) * (pointA[0] - pointB[0]))


if __name__ == '__main__':
    #filename = 'sample.jpg'
    filename = 'IMG_2713.jpg'
    #filename = 'IMG_2713gedreht.JPG'
    image = cv2.imread(filename,-1)
    
    #rows,cols, _ = image.shape

    #M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
    #image = cv2.warpAffine(image,M,(cols,rows))

    #image2 = cv2.resize(image, (200, 200))
    image2 = cv2.resize(image, (800, 800))
    gray = cv2.cvtColor(image2,cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,100,200)
    cv2.imshow("canny", edges)
    edges, contours, hierachy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    mc = [ (moments['m10']/moments['m00'], moments['m01']/moments['m00']) if moments['m00'] != 0 else (float('inf'), float('inf')) 
        for moments in  (cv2.moments(contour, False) for contour in contours) ]
    
    def getDept(index):
        count = 0;
        while hierachy[0][index][2] >= 0:
            index = hierachy[0][index][2];
            count +=1
        return count
    
    marks = [i for i in xrange(len(hierachy[0])) if getDept(i) >= 5]
    
    print marks
    if len(marks) >= 3:
        marks = marks[0:3]

        distance_iterator = ((euklied_distance(mc[bottomleft], mc[topright]), (topleft, topright, bottomleft)) 
                            for bottomleft, topleft, topright in itertools.permutations(marks) if 0 < cross_product(mc[bottomleft], mc[topleft], mc[topright]))
        _ , point_list = max(distance_iterator)
        topleft, topright, bottomleft = point_list

        slope = line_slope(mc[bottomleft], mc[topright])

        relevant = [(contours[pattern], mc[pattern]) for pattern in  point_list]
        
        if all(cv2.contourArea(contour) > 10 for contour, _ in relevant):
            def corners_iterable():
                for contour, point in relevant:
                    x, y, w, h = cv2.boundingRect(contour)
                    if abs(slope) > 5:
                        a, b, c, d = [x,y], [x + w, y], [x + w, y + h], [x, y + h]
                        categories = (((line_distance(c, a, [x_contour_point,y_contour_point]), line_distance(b, d, [x_contour_point,y_contour_point])), contour_point) for [[x_contour_point,y_contour_point]] in contour)
                    
                    else:
                        x_middle, y_middle = x + w / 2, y + h / 2
                        categories = ((y_contour_point > y_middle, x_contour_point > x_middle, euklied_distance(point, [x_contour_point,y_contour_point]), [x_contour_point,y_contour_point]) for  [[x_contour_point,y_contour_point]]  in contour)
                        #a, b, c, d = [x, y + h / 2], [x + w / 2 ,y], [x + w, y + h / 2], [x + w / 2, y + h]
                
                    grouped_corners = itertools.groupby(sorted(categories), operator.itemgetter(0,1))
                    max_differences = (max(values, key = operator.itemgetter(2)) for categorie, values in grouped_corners)
                    corners_coordinates = [coordinates for (_ , _ , _ , coordinates) in max_differences]
                    yield corners_coordinates
            ccl = numpy.array(list(corners_iterable()))
            print ccl
            #ccl = numpy.roll(ccl,2,1)
            print ccl
            
            tl = 0
            tr = 1
            bl = 2
            br = 3

            
            topright_dif =   ccl[tr][tr] - ccl[tr][br]
            bottomleft_dif = ccl[bl][bl] - ccl[bl][br]   
            
            if numpy.linalg.det([topright_dif, bottomleft_dif]) != 0:
                t = numpy.linalg.det([ccl[bl][bl] - ccl[tr][tr], bottomleft_dif]) / numpy.linalg.det([topright_dif, bottomleft_dif])
                bottomright_bottomright = ccl[tr][tr] + t * topright_dif
                print bottomright_bottomright

                source = numpy.array([ccl[tl][tl], ccl[tr][tr], bottomright_bottomright, ccl[bl][bl]], dtype = "float32")

                topleft_width  = (numpy.linalg.norm(ccl[tl][tl]-ccl[tl][tr]) + numpy.linalg.norm(ccl[tl][bl]-ccl[tl][br]) + numpy.linalg.norm(ccl[tr][tl]-ccl[tr][tr]) + numpy.linalg.norm(ccl[tr][bl]-ccl[tr][br])) / 4
                top_width =      (numpy.linalg.norm(ccl[tl][tl]-ccl[tr][tr]) + numpy.linalg.norm(ccl[tl][bl]-ccl[tr][br])) / 2
                topleft_heigth = (numpy.linalg.norm(ccl[tl][tl]-ccl[tl][bl]) + numpy.linalg.norm(ccl[tl][tr]-ccl[tl][br]) + numpy.linalg.norm(ccl[bl][tl]-ccl[bl][bl]) + numpy.linalg.norm(ccl[bl][tr]-ccl[bl][br])) / 4
                left_heigth =    (numpy.linalg.norm(ccl[tl][tl]-ccl[bl][bl]) + numpy.linalg.norm(ccl[tl][tr]-ccl[bl][br])) / 2
                pixelcount_horizontal = top_width / topleft_width * 7
                pixelcount_vertical = left_heigth / topleft_heigth * 7
                

                pixelmean = int(round((pixelcount_horizontal + pixelcount_vertical) / 2))
                print pixelmean

                width = heigth = pixelmean * 8
                destination = numpy.array([(0, 0), (width, 0), (width, heigth), (0, heigth)], dtype = "float32")
    
                print source, destination
                warp_matrix = cv2.getPerspectiveTransform(source, destination);
                qr_raw = cv2.warpPerspective(image2, warp_matrix, (width, heigth));	
                qr_gray = cv2.cvtColor(qr_raw,cv2.COLOR_RGB2GRAY);
                _ ,qr_thres = cv2.threshold(qr_gray, 127, 255, cv2.THRESH_BINARY);
                qr_small = cv2.resize(qr_thres, (pixelmean, pixelmean))
                a ,qr_thres_small = cv2.threshold(qr_small, 127, 255, cv2.THRESH_BINARY);
                #cv2.imshow("qrtres", qr_thres)# cv2.resize(qr_thres, (10*width, 10*heigth)))
                #cv2.imshow("small", qr_thres_small)
                cv2.imshow("big", cv2.resize(qr_thres_small, (width, heigth), interpolation = cv2.INTER_NEAREST))

                # decoding
                #qr = qrtools.QR()
                #qr.cecode(qr_thres_small)
                #print qr.data

            #    return p + t*r

            
            #for contour, point in relevant:
            #    x, y, w, h = cv2.boundingRect(contour)
            #    if abs(slope) > 5:
            #        a, b, c, d = [x,y], [x + w, y], [x + w, y + h], [x, y + h]
            #        categories = (((line_distance(c, a, [x_contour_point,y_contour_point]), line_distance(b, d, [x_contour_point,y_contour_point])), contour_point) for [[x_contour_point,y_contour_point]] in contour)
                    
            #    else:
            #        x_middle, y_middle = x + w / 2, y + h / 2
            #        categories = (((x_contour_point < x_middle, y_contour_point < y_middle), (x_contour_point,y_contour_point)) for  [[x_contour_point,y_contour_point]]  in contour)
            #        #a, b, c, d = [x, y + h / 2], [x + w / 2 ,y], [x + w, y + h / 2], [x + w / 2, y + h]
                
            #    grouped_corners = itertools.groupby(sorted(categories))
            #    max_differences = [max(values, key = lambda x : euklied_distance(point, (x[1][0],x[1][0])) ) for categorie, values in grouped_corners]            


                #print categorie
                #print list(max_differences)
                #print '======================='    
                #print categories
                #print '======================='    
                #grouped = sorted(categories, operator.gt)#, lambda x : x[0])
                #print abs(slope) > 5, a, b, c, d
                #print contour
                #print[((line_distance(c, a, contour_point[0]), line_distance(b, d, contour_point[0])),
                #       (cross_product(c, a, contour_point[0]), cross_product(b, d, contour_point[0])), contour_point) for contour_point in contour]
                
                
                #grouped = sorted(categories, lambda x, y: x[0] > y[0]) # key = lambda x : x[0])#
                #print list(grouped)
                #print list((sorted(group)))
  
                #print list(group)
             

        #dist = line_distance(mc[bottomleft], mc[topright], mc[topleft])	
        
    #if False:
    #    bottom = median1;
    #    right = median2;
    #elif slope < 0 and dist < 0:
    #    bottom = median1;
    #    right = median2;
    #    orientation = CV_QR_NORTH;
    #elif slope > 0 and dist < 0:
    #    right = median1;
    #    bottom = median2;
    #    orientation = CV_QR_EAST;
    #elif slope < 0 and dist > 0:
    #    right = median1;
    #    bottom = median2;
    #    orientation = CV_QR_SOUTH;
    #elif slope > 0 and dist > 0 :
    #    bottom = median1;
    #    right = median2;
    #    orientation = CV_QR_WEST;
    #print mu
    #print mc
    #print contours
    #print hierachy
    #print image
    ##image3 = cv2.
    ##image, image, ´Size(200, 200));
    #pass

    cv2.waitKey(0)
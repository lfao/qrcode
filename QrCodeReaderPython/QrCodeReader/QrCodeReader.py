import numpy
import cv2
import itertools
import functools
import operator
import math
import qrtools


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
    image = cv2.imread(filename,-1)
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
            corners_coordinates_list = list(corners_iterable())
            
            print corners_coordinates_list
            
            [[topleft_topleft,topleft_topright,topleft_bottomleft,_], [topright_topleft, topright_topright, _ ,topright_bottomright ],[_ ,bottomleft_topleft ,bottomleft_bottomleft, bottomleft_bottomright]] = corners_coordinates_list
            tl = 0
            tr = 1
            bl = 2
            br = 3
            ccl = corners_coordinates_list
            
            print topright_topright, topright_bottomright ,bottomleft_bottomleft, bottomleft_bottomright

            def cross((v1x, v1y), (v2x, v2y)):
                return float(v1x) * float(v2y) - float(v1y) * float(v2x)
            def sub((v1x, v1y), (v2x, v2y)):
                return v1x - v2x, v1y - v2y
            def add((v1x, v1y), (v2x, v2y)):
                return v1x + v2x, v1y + v2y
            def neg((v1x, v1y)):
                return -v1x, -v1y
            def mult(t, (v1x, v1y)):
                return t * v1x, t * v1y 
            
            
            topright_dif =   sub(topright_topright, topright_bottomright)
            bottomleft_dif = sub(bottomleft_bottomleft, bottomleft_bottomright)
            #bottomleft_dif_neg
           
            print topright_bottomright, topright_topright, bottomleft_bottomright, bottomleft_bottomleft
            
            if cross(topright_dif, bottomleft_dif) != 0:
                t = cross(sub(bottomleft_bottomleft, topright_topright),bottomleft_dif) / cross(topright_dif, bottomleft_dif)
                bottomright_bottomright = add(topright_topright, mult(t, topright_dif))
                print bottomright_bottomright

                source = numpy.array([topleft_topleft, topright_topright, bottomright_bottomright, bottomleft_bottomleft], dtype = "float32")

                topleft_width  = (euklied_distance(ccl[tl][tl],ccl[tl][tr]) + euklied_distance(ccl[tl][bl],ccl[tl][br]) + euklied_distance(ccl[tr][tl],ccl[tr][tr]) + euklied_distance(ccl[tr][bl],ccl[tr][br])) / 4
                top_width =   (euklied_distance(ccl[tl][tl],ccl[tr][tr]) + euklied_distance(ccl[tl][bl],ccl[tr][br])) / 2
                topleft_heigth = (euklied_distance(ccl[tl][tl],ccl[tl][bl]) + euklied_distance(ccl[tl][tr],ccl[tl][br]) + euklied_distance(ccl[bl][tl],ccl[bl][bl]) + euklied_distance(ccl[bl][tr],ccl[bl][br])) / 4
                left_heigth = (euklied_distance(ccl[tl][tl],ccl[bl][bl]) + euklied_distance(ccl[tl][tr],ccl[bl][br])) / 2
                pixelcount_horizontal = top_width / topleft_width * 7
                pixelcount_vertical = left_heigth / topleft_heigth * 7
                

                pixelmean = int(round((pixelcount_horizontal + pixelcount_vertical) / 2))
                print pixelmean



                width = heigth = pixelmean * 10
                destination = numpy.array([(0, 0), (width, 0), (width, heigth), (0, heigth)], dtype = "float32")
    





                print source, destination
                warp_matrix = cv2.getPerspectiveTransform(source, destination);
                qr_raw = cv2.warpPerspective(image2, warp_matrix, (width, heigth));
                #qr = cv2.copyMakeBorder(qr_raw, 10, 10, 10, 10,cv2.BORDER_CONSTANT, value = [255, 255, 255]);	
                qr = qr_raw
                qr_gray = cv2.cvtColor(qr,cv2.COLOR_RGB2GRAY);
                _ ,qr_thres = cv2.threshold(qr_gray, 127, 255, cv2.THRESH_BINARY);
                qr_small = cv2.resize(qr_thres, (pixelmean, pixelmean))
                a ,qr_thres_small = cv2.threshold(qr_small, 127, 255, cv2.THRESH_BINARY);
                cv2.imshow("qrtres", qr_thres)# cv2.resize(qr_thres, (10*width, 10*heigth)))
                cv2.imshow("small", qr_thres_small)
                cv2.imshow("big", cv2.resize(qr_thres_small, (width, heigth), interpolation = cv2.INTER_NEAREST))

                # decoding
                qr = qrtools.QR()
                qr.decode(qr_thres_small)
                print qr.data
                
            #def getIntersectionPoint(a1, a2, b1, b2):
            #    p = a1
            #    q = b1
            #    r = a2-a1
            #    s = b2-b1

            #    if(cross(r,s) == 0): return None

            #    t = cross(q-p,s) / cross(r,s)

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
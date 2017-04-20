import numpy
import cv2
import itertools
import operator

def extract_qr_bin(image, output_size = None):
    '''
    Extract a matrix of boolean values for a QR Code.
    Every boolean is associated to one point of the QR code.
    Keyword arguments:
    image -- The index in hierachy to check
    output_size (Default None) The size of the picture in the output for debug reasons. None if no pictures should be printed on the screen
    Returns:
    A numpy matrix with boolean value for every point in the QR code
    '''
    #Definitions TopLeft, TopRight, BottomLeft, BottomRight are required for indexing of Finder Pattern itself or Finder Pattern corners
    TL, TR, BL, BR = range(4) 

    image_resized = cv2.resize(image, (800, 800))
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(image_gray, 100, 200)
    _, contours, [hierachy] = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def get_dept(index):
        '''
        Looks for the amount of childs in hierachy for a certain index
        Keyword arguments:
        index -- The index in hierachy to check
        Returns:
        The amount of childs of the selected index
        '''
        return get_dept(hierachy[index][2]) + 1 if hierachy[index][2] >= 0 else 0

    # finding Finder Pattern which is a 5 times nested contour
    marks = [i for i in xrange(len(hierachy)) if get_dept(i) == 5]

    if len(marks) != 3: # check if 3 and only 3 finder pattern have been found
        print("Detected {} Finder Pattern. Exact 3 are required!".format(len(marks)))
        return None

   # checking if size is enough for getting good values
    if any(cv2.contourArea(contours[mark]) < 10 for mark in marks):
        print("Some of the detected Finder Pattern are to small!")
        return None    

    # calculating the center of the contour of each pattern
    moments_list = (cv2.moments(contours[mark]) for mark in marks)
    unsorted_center_list = numpy.array([ (moments['m10']/moments['m00'], moments['m01']/moments['m00']) 
                                   if moments['m00'] != 0 else (float('inf'), float('inf')) for moments in moments_list])

    # matching the Finter Pattern to the corners  TL, TR, BL
    distance_patternlist_tuple_list = ((numpy.linalg.norm(unsorted_center_list[pattern_triple[BL]] - unsorted_center_list[pattern_triple[TR]]), pattern_triple) # generating a tuple of distance and the patterntriple
            for pattern_triple in itertools.permutations(range(3)) # iterating through permutations of possible matchings
            if 0 < numpy.cross(unsorted_center_list[pattern_triple[TR]] - unsorted_center_list[pattern_triple[TL]], # filtering for clockwise matchings (TL TR BL)
                               unsorted_center_list[pattern_triple[BL]] - unsorted_center_list[pattern_triple[TL]])) 
                                # https://math.stackexchange.com/questions/285346/why-does-cross-product-tell-us-about-clockwise-or-anti-clockwise-rotation

    # take the pattern tripple of the one with the greatest distance between BottomLeft and TopRight
    _ , pattern_triple = max(distance_patternlist_tuple_list) 

    pattern_contour_list = (contours   [marks   [pattern]] for pattern in pattern_triple)
    pattern_center_list  = [unsorted_center_list[pattern]  for pattern in pattern_triple]

    # calculating horizontal and vertical vectors for the alligned qr code
    horizontal_vector = pattern_center_list[TR] - pattern_center_list[TL]
    verticial_vector =  pattern_center_list[BL] - pattern_center_list[TL]
    
    # extracting 4 corners for each pattern
    def pattern_corner_generator():
        '''
        Generates a list of 4 corner for each pattern
        Returns:
        A generator for the lists of corners
        '''
        for contour, center in itertools.izip(pattern_contour_list , pattern_center_list):
            # creating triples of:  
            #   an tuple of bools indicating if they are up or down, left or right. Sorting these ascending will cause the order TL, TR, BL, BR
            #       therefore the sign of the crossproduct of two vectors is used: http://stackoverflow.com/questions/3838319/how-can-i-check-if-a-point-is-below-a-line-or-not
            #   the distance between this contour point and the Finder Pattern center 
            #   the contour point
            categorie_distance_point_triple_list = (((numpy.cross(horizontal_vector, contour_point - center) > 0, numpy.cross(verticial_vector, contour_point - center) < 0), 
                                                      numpy.linalg.norm(contour_point - center), contour_point) for [contour_point] in contour)
            
            # sorting and matching the triples into 4 groups of each corner by using the bool tuple
            # (false, false) <=> TL vs. (false, true) <=> TR vs. (true, false) <=> BL vs (true, true) <=> BR
            corner_selection_tuple_list = itertools.groupby(sorted(categorie_distance_point_triple_list, key = operator.itemgetter(0)), operator.itemgetter(0))
            
            # taking the contour point with the longest distance to the center for each corner. 
            # The key of each categorie is not required since the order is implicit like the definitions of TL TR BL BR
            corner_points_triple_list = (max(values, key = operator.itemgetter(1)) for _ , values in corner_selection_tuple_list)

            # remove the bool tuple and the distance and store only the corner coordinates in a list
            corner_coordinate_list = [coordinates for _ , _  , coordinates in corner_points_triple_list]
            yield corner_coordinate_list
    
    # creating a 2D numpy matrix whose first index is the Finder Pattern and the second index is the corner 
    # [patternindex][cornerindex]
    # patternindices are TL, TR, BL
    # conrerindices  are TL, TR, BL, BR 
    pattern_corner_list = numpy.array(list(pattern_corner_generator()))

    # extrapolation of the bottom right corner http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    tr_r_dif_vertical = pattern_corner_list[TR][TR] - pattern_corner_list[TR][BR]
    bl_b_dif_horizontal = pattern_corner_list[BL][BL] - pattern_corner_list[BL][BR]   
    t = float(numpy.cross(pattern_corner_list[BL][BL] - pattern_corner_list[TR][TR], bl_b_dif_horizontal)) / numpy.cross(tr_r_dif_vertical, bl_b_dif_horizontal)
    br_br = pattern_corner_list[TR][TR] + t * tr_r_dif_vertical

    # defining the warp source quadrangle
    source = numpy.array([pattern_corner_list[TL][TL], pattern_corner_list[TR][TR], br_br, pattern_corner_list[BL][BL]], dtype = "float32")
    
    # calculating the number of pixels in the clean qr code
    pattern_average = numpy.mean([numpy.linalg.norm(pattern_corner_list[i][j]-pattern_corner_list[i][k]) for i in xrange(3) for j, k in [(TL,TR),(BL,BR),(TL,BL),(TR,BR)]])
    size_average    = numpy.mean([numpy.linalg.norm(pattern_corner_list[TL][TL]-pattern_corner_list[TR][TR]),  
                                  numpy.linalg.norm(pattern_corner_list[TL][BL]-pattern_corner_list[TR][BR]), 
                                  numpy.linalg.norm(pattern_corner_list[TL][TL]-pattern_corner_list[BL][BL]), 
                                  numpy.linalg.norm(pattern_corner_list[TL][TR]-pattern_corner_list[BL][BR])])
    pixelcount = int(round(size_average / pattern_average * 7)) # the width and the heigth of Finder Pattern is 7. Use the rule of three

    
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

    if output_size:
        cv2.imshow("resized",cv2.resize(image_resized, (output_size,output_size)))
        cv2.imshow("gray",cv2.resize(image_gray , (output_size,output_size)))
        cv2.imshow("canny", cv2.resize(edges, (output_size,output_size)))
        cv2.imshow("bigqr_nottresholded", bigqr_nottresholded)
        cv2.imshow("bigqr", bigqr)
        cv2.imshow("qr small", qr, )
        cv2.imshow("qr", cv2.resize(qr, (temp_warp_size, temp_warp_size), interpolation = cv2.INTER_NEAREST))
        cv2.imwrite("output.jpg",qr)
    return data


if __name__ == '__main__':

    #filename = '45degree.jpg' # easy
    #filename = 'IMG_2713.jpg' # chair
    filename = 'IMG_2717.JPG' # wall
    #filename = 'IMG_2716.JPG' # keyboard , little extrapolation error
    #filename = 'IMG_2712.JPG' # wall, not flat, very high slope , little warping error    

    image = cv2.imread(filename,-1)
    binary = extract_qr_bin(image, 400)
    print binary

    cv2.waitKey(0)



import numpy
import cv2
import itertools
import operator

def extract_matrix(image, output_size = None):
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

    image_resized = cv2.resize(image, (600, 600)) # 12 ok
    #image_resized = image;
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
        # the child is stored in hierachy at inner index = 2, it is negative if it has no child
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
    distance_patternlist_tuple_list = ((numpy.linalg.norm(unsorted_center_list[patternindex_triple[BL]] - unsorted_center_list[patternindex_triple[TR]]), patternindex_triple) # generating a tuple of distance and the patterntriple
            for patternindex_triple in itertools.permutations(range(3)) # iterating through permutations of possible matchings
            if 0 < numpy.cross(unsorted_center_list[patternindex_triple[TR]] - unsorted_center_list[patternindex_triple[TL]], # filtering for clockwise matchings (TL TR BL)
                               unsorted_center_list[patternindex_triple[BL]] - unsorted_center_list[patternindex_triple[TL]])) 
                                # https://math.stackexchange.com/questions/285346/why-does-cross-product-tell-us-about-clockwise-or-anti-clockwise-rotation

    # take the pattern tripple of the one with the greatest distance between BottomLeft and TopRight
    _ , patternindex_triple = max(distance_patternlist_tuple_list) 

    # Reordering the and selecting the reqired contours and centers
    pattern_contour_list = (contours   [marks   [pattern]] for pattern in patternindex_triple)
    pattern_center_list  = [unsorted_center_list[pattern]  for pattern in patternindex_triple]

    # calculating horizontal and vertical vectors for the alligned qr code
    # this does not reqire to be exact
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
    
    # calculating the number of pixels in the clean qr code. This must be very exact
    pattern_average = numpy.mean([numpy.linalg.norm(pattern_corner_list[i] [j]  - pattern_corner_list[i] [k] ) for i in xrange(3) for j, k in [(TL,TR),(BL,BR),(TL,BL),(TR,BR)]])
    size_average    = numpy.mean([numpy.linalg.norm(pattern_corner_list[TL][TL] - pattern_corner_list[TR][TR]),  
                                  numpy.linalg.norm(pattern_corner_list[TL][BL] - pattern_corner_list[TR][BR]), 
                                  numpy.linalg.norm(pattern_corner_list[TL][TL] - pattern_corner_list[BL][BL]), 
                                  numpy.linalg.norm(pattern_corner_list[TL][TR] - pattern_corner_list[BL][BR])])
    
    pixelestimated = size_average / pattern_average * 7 # the width and the heigth of Finder Pattern is 7. Use the rule of three
    pixelcount = int(round((pixelestimated - 17) / 4)) * 4 + 17 # only pixelcounts of 4 * Version + 17 are allowed => round to this number
    pixel_lenght = size_average / pixelcount

    
    # extrapolation of the bottom right corner http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    # This must be very exact
    tr_r_dif_vertical = pattern_corner_list[TR][TR] - pattern_corner_list[TR][BR]
    bl_b_dif_horizontal = pattern_corner_list[BL][BL] - pattern_corner_list[BL][BR]   
    t = float(numpy.cross(pattern_corner_list[BL][BL] - pattern_corner_list[TR][TR], bl_b_dif_horizontal)) / numpy.cross(tr_r_dif_vertical, bl_b_dif_horizontal)
    br_br = pattern_corner_list[TR][TR] + t * tr_r_dif_vertical     
    
    found_alginment = False
    if (pixelcount - 17) / 4 > 1: # check if the QR code has a version greater 1. Then it has a alignment pattern at the bottom right corner too. Use this for a better extrapolation
        # trying to find this alignment pattern and improve the value.

        alignment_area_delta = (pixel_lenght) * 6 # The width and height of the area for searching the contour of the finderpattern will be 2 * aligment_area_delta

        # calculating the middle of the QR code with the centers of the pattern TR and BL 
        # calculating the vector from pattern_corner_list[TL][BR] to the middle of the QR code
        # adding this vector two times to pattern_corner_list[TL][BR] to reach the pattern_corner_list[BR][TL] (which values do not exist yet)
        # this coordinates are a half pixel away from the center. the area of searching this center should be big enought to handle this offset
        estimated_alignment_center = (pattern_center_list[TR] + pattern_center_list[BL] - pattern_corner_list[TL][BR])

        # getting the start and the end of the slice for each coordinate of the alignment center
        # (numpy style adding a 2 element column vector to a 2 elment row vector will create a matrix with 4 elements) 
        slice_coordinates = (numpy.transpose([estimated_alignment_center]) + [-alignment_area_delta, alignment_area_delta]).astype(int) 

        slice_coordinates[:,0] = numpy.maximum(slice_coordinates[:,0], [0,0])
        slice_coordinates[:,1] = numpy.minimum(slice_coordinates[:,1], edges.shape)

        area = (slice(*slice_coordinates[1]), slice(*slice_coordinates[0])) # get slice obejects for the search area
        # finding contours in the selected area
        _, alignment_contours, [alignment_hierachy] = cv2.findContours(edges[area], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        def alignment_get_dept(index):
            '''
            Looks for the amount of childs in alignment_hierachy for a certain index
            Keyword arguments:
            index -- The index in hierachy to check
            Returns:
            The amount of childs of the selected index
            '''
            # the child is stored in hierachy at inner index = 2, it is negative if it has no child
            return alignment_get_dept(alignment_hierachy[index][2]) + 1 if alignment_hierachy[index][2] >= 0 else 0 

        # find a 3 times nested contour which is the alignment pattern
        alignment_index = [i for i in xrange(len(alignment_hierachy)) if alignment_get_dept(i) == 3]

        if 1 == len(alignment_index): # check whether exact one pattern has been found
            # make a better extrapolation of the bottom right corner

            # calculate the center of the alignment pattern
            alignment_moments = cv2.moments(alignment_contours[alignment_index[0]])
            alignment_center = slice_coordinates[:,0] + [alignment_moments['m10']/alignment_moments['m00'], alignment_moments['m01']/alignment_moments['m00']]

            found_alginment = True


        def get_error(rho, theta, pattern, corners):
            orthogonal = numpy.array([numpy.cos(theta), numpy.sin(theta)])
            return abs(numpy.linalg.norm([numpy.dot(orthogonal, pattern_corner_list[pattern][corner]) - rho for corner in corners]))
  

        #print size_average
        fieldsize = size_average / 2.5
    
        houghfield = numpy.zeros(edges.shape, dtype = edges.dtype)
        houghfield[int(estimated_alignment_center[1] - fieldsize) : int(estimated_alignment_center[1] + fieldsize), int(estimated_alignment_center[0] - fieldsize): int(estimated_alignment_center[0] + fieldsize)] = (
             edges[int(estimated_alignment_center[1] - fieldsize) : int(estimated_alignment_center[1] + fieldsize), int(estimated_alignment_center[0] - fieldsize): int(estimated_alignment_center[0] + fieldsize)])
        lines = cv2.HoughLines(houghfield,1,numpy.pi/180, int(size_average / 6))
        if lines is not None:
            bottom_distance , bottom_line = min((get_error(rho, theta, BL, [BL, BR]), (rho, theta)) for [[rho, theta]] in lines)
            right_distance  , right_line  = min((get_error(rho, theta, TR, [TR, BR]), (rho, theta)) for [[rho, theta]] in lines)

            valid =  numpy.all(numpy.array([bottom_distance , right_distance]) / pixel_lenght < 0.5)

            img = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
            #for [[rho,theta]] in lines:
            #    if abs(theta - angle_hor) < delta or abs(theta - angle_ver) < delta or True:
                #if 0.8 < theta < 1.2 or 2 < theta < 2.5:

            for [rho,theta] in [bottom_line, right_line]:
                #print theta #/ numpy.pi * 180
                a = numpy.cos(theta)
                b = numpy.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(img,(x1,y1),(x2,y2),(0,255 * (not valid),255 * valid),1)


            cv2.imshow('houghlines',img)
            #cv2.imshow('reduced',houghfield)

    
    #
    
    # defining the warp destination square which is 8*8 times the number of pixels in the clean qr code
    temp_pixel_size = 8
    temp_warp_size = pixelcount * temp_pixel_size
    boarder = temp_pixel_size * 0

    # defining the warp source and destination quadrangle
    if found_alginment: # if we found the alignment pattern at bottom right, use this
        source = numpy.array([pattern_corner_list[TL][TL], pattern_corner_list[TR][TR], alignment_center, pattern_corner_list[BL][BL]], dtype = "float32")
        destination = numpy.array([(boarder, boarder), (temp_warp_size + boarder, boarder), (temp_warp_size + boarder - temp_pixel_size * 6.5, temp_warp_size + boarder - temp_pixel_size * 6.5), (0 + boarder, temp_warp_size + boarder)], dtype = "float32")
    else: # otherwise use the extrapolated bottom right corner
        source = numpy.array([pattern_corner_list[TL][TL], pattern_corner_list[TR][TR], br_br, pattern_corner_list[BL][BL]], dtype = "float32")
        destination = numpy.array([(boarder, boarder), (temp_warp_size + boarder, boarder), (temp_warp_size + boarder, temp_warp_size + boarder), (0 + boarder, temp_warp_size + boarder)], dtype = "float32")
   
    # doing the warping and thresholding
    warp_matrix = cv2.getPerspectiveTransform(source, destination);
    bigqr_nottresholded = cv2.warpPerspective(image_gray, warp_matrix, (temp_warp_size + 2 * boarder, temp_warp_size + 2 * boarder));	

    bigqr = cv2.adaptiveThreshold(bigqr_nottresholded,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,temp_pixel_size * 10 + 1,2)
    #_ ,bigqr = cv2.threshold(bigqr_nottresholded, 127, 255, cv2.THRESH_BINARY);   

    # resizing to the real amount of pixels and thresholding again
    qr_notresholded = cv2.resize(bigqr[boarder : temp_warp_size + boarder, boarder : temp_warp_size + boarder], (pixelcount, pixelcount))
    a ,qr = cv2.threshold(qr_notresholded, 127, 255, cv2.THRESH_BINARY);
    
    # extracting the data
    inverted_data = numpy.asarray(qr, dtype="bool")

    if output_size:
        cv2.imshow("resized",cv2.resize(image_resized, (output_size,output_size)))
        cv2.imshow("gray",cv2.resize(image_gray , (output_size,output_size)))
        cv2.imshow("canny", cv2.resize(edges, (output_size,output_size)))
        if (pixelcount - 17) / 4 > 1: 
            cv2.imshow("alignment pattern", edges[area]) 

        cv2.imshow("bigqr_nottresholded", bigqr_nottresholded)
        cv2.imshow("bigqr", bigqr)
        cv2.imshow("qr small", qr, )
        cv2.imshow("qr", cv2.resize(qr, (temp_warp_size, temp_warp_size), interpolation = cv2.INTER_NEAREST))
        cv2.imwrite("output.jpg",qr)
    return numpy.logical_not(inverted_data)
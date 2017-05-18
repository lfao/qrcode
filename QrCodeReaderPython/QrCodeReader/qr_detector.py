from __future__ import division
from builtins import range

import numpy
import cv2
import itertools
import operator

class QrDetectorError(Exception):
	pass

def extract_matrix(image, output_size=None):
	'''
	Extracts a matrix of boolean values for a qr code.
	Every boolean is associated to one point of the qr code.
	Keyword arguments:
	image -- The opencv image
	output_size (Default None) The size of the picture in the output for debug reasons. None, if no pictures should be printed on the screen
	Returns:
	A numpy matrix with a boolean value for every point of the qr code
	'''
	# Definitions TopLeft, TopRight, BottomLeft, BottomRight are required for
	# indexing of finder pattern itself or finder pattern corners
	TL, TR, BL, BR = range(4)

	image_resized = cv2.resize(image, (600, 600))
	#image_resized = image;
	image_gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
	edges = cv2.Canny(image_gray, 100, 200)

	_, contours, [hierarchy] = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	def get_dept(index):
		'''
		Looks for the amount of children in hierarchy for a certain index
		Keyword arguments:
		index -- The index in hierarchy to check
		Returns:
		The amount of children of the selected index
		'''
		# the child is stored in hierarchy at inner index = 2
		# it is negative, if it has no child
		return get_dept(hierarchy[index][2]) + 1 if hierarchy[index][2] >= 0 else 0 

	# finding finder pattern which is a 5 times nested contour
	marks = [i for i in range(len(hierarchy)) if get_dept(i) == 5]

	if len(marks) != 3: # check if 3 and only 3 finder pattern have been found
		if output_size is not None:
			img = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
			selected_contours = [contours[i] for i in marks]
			cv2.drawContours(img, selected_contours, -1, (0,255,0), 1)
			cv2.imshow('resized',cv2.resize(image_resized, (output_size,output_size)))
			cv2.imshow('gray',cv2.resize(image_gray , (output_size,output_size)))
			cv2.imshow('pattern', img)
		raise QrDetectorError('Detected {} Finder Pattern. Exact 3 are required!'.format(len(marks)))

   # checking if size is enough for getting good values
	if any(cv2.contourArea(contours[mark]) < 10 for mark in marks):
		raise QrDetectorError('Some of the detected Finder Pattern are to small!')

	# calculating the center of the contour of each pattern
	moments_list = (cv2.moments(contours[mark]) for mark in marks)
	unsorted_center_list = [numpy.array([moments['m10'], moments['m01']]) / moments['m00'] for moments in moments_list]

	# matching the finder pattern to the corners TL, TR, BL
	distance_patternlist_tuple_list = ((numpy.linalg.norm(unsorted_center_list[patternindex_triple[BL]] - unsorted_center_list[patternindex_triple[TR]]), patternindex_triple) # generating a tuple of distance and the pattern triple
			for patternindex_triple in itertools.permutations(range(3)) # iterating through permutations of possible matchings
			if 0 < numpy.cross(unsorted_center_list[patternindex_triple[TR]] - unsorted_center_list[patternindex_triple[TL]], # filtering for clockwise matchings (TL TR BL)
							   unsorted_center_list[patternindex_triple[BL]] - unsorted_center_list[patternindex_triple[TL]])) 
								# https://math.stackexchange.com/questions/285346/why-does-cross-product-tell-us-about-clockwise-or-anti-clockwise-rotation

	# taking the pattern triple of the one with the greatest distance between
	# BottomLeft and TopRight
	_, patternindex_triple = max(distance_patternlist_tuple_list) 

	# Reordering and selecting the required contours and centers
	pattern_contour_list = (contours[marks[pattern]] for pattern in patternindex_triple)
	pattern_center_list = [unsorted_center_list[pattern]  for pattern in patternindex_triple]

	# calculating horizontal and vertical vectors for the aligned qr code
	# this does not require to be exact
	horizontal_vector = pattern_center_list[TR] - pattern_center_list[TL]
	verticial_vector = pattern_center_list[BL] - pattern_center_list[TL]
	
	# extracting 4 corners for each pattern
	def pattern_corner_generator():
		'''
		Generates a list of 4 corners for each pattern
		Returns:
		A generator for the lists of corners
		'''
		for contour, center in zip(pattern_contour_list , pattern_center_list):
			# creating triples of:
			#   a tuple of booleans indicating if they are up or down, left or
			#       right.  Sorting these ascending will cause the order TL, TR,
			#       BL, BR
            #   distance to the center point
			#   the contour point
			# for the tuples of booleans the sign of the crossproduct of two
			# vectors is used:
			# http://stackoverflow.com/questions/3838319/how-can-i-check-if-a-point-is-below-a-line-or-not
			# the distance between this contour point and the finder pattern

			categorie_distance_point_triple_list = (((numpy.cross(horizontal_vector, contour_point - center) > 0, numpy.cross(verticial_vector, contour_point - center) < 0), 
													  numpy.linalg.norm(contour_point - center), contour_point) for [contour_point] in contour)
			
			# sorting and matching the triples into 4 groups of each corner by
			# using the boolean tuple
			# (false, false) <=> TL vs.  (false, true) <=> TR 
			# vs.  (true, false) <=> BL vs (true, true) <=> BR
			corner_selection_tuple_list = itertools.groupby(sorted(categorie_distance_point_triple_list, key = operator.itemgetter(0)), operator.itemgetter(0))
			
			# taking the contour point with the longest distance to the center
			# for each corner.
			# The key of each category is not required since the order is
			# implicit like the definitions of TL TR BL BR
			corner_points_triple_list = (max(values, key = operator.itemgetter(1)) for _ , values in corner_selection_tuple_list)

			# removing the boolean tuple and the distance and storing only the corner
			# coordinates in a list
			corner_coordinate_list = [coordinates for _ , _  , coordinates in corner_points_triple_list]
			yield corner_coordinate_list
	
	# creating a 2D numpy matrix whose first index is the finder pattern and
	# the second index is the corner
	# [pattern index][corner index]
	# pattern indices are TL, TR, BL
	# corner indices are TL, TR, BL, BR
	pattern_corner_list = numpy.array(list(pattern_corner_generator()))
	
	# calculating the number of pixels in the clean qr code 
	# This must be very exact
	pattern_average = numpy.mean([numpy.linalg.norm(pattern_corner_list[i][j] - pattern_corner_list[i][k]) for i in range(3) for j, k in [(TL,TR),(BL,BR),(TL,BL),(TR,BR)]])
	size_average = numpy.mean([numpy.linalg.norm(pattern_corner_list[TL][TL] - pattern_corner_list[TR][TR]),  
								  numpy.linalg.norm(pattern_corner_list[TL][BL] - pattern_corner_list[TR][BR]), 
								  numpy.linalg.norm(pattern_corner_list[TL][TL] - pattern_corner_list[BL][BL]), 
								  numpy.linalg.norm(pattern_corner_list[TL][TR] - pattern_corner_list[BL][BR])])
	
	estimated_pixel_count = size_average / pattern_average * 7 # the width and the height of finder pattern is 7.  Using the rule of three
	version = int(round((estimated_pixel_count - 17) / 4)) # only estimated_pixel_count of 4 * Version + 17 are allowed => rounding to this number
	if not 0 < version <= 40:
		raise QrDetectorError('Calculated version {} is not between 1 and 40!'.format(version))

	pixelcount = version * 4 + 17 
	pixel_lenght = size_average / pixelcount

	# defining the warp destination square which is 8*8 times the number of
	# pixels in the clean qr code
	temp_pixel_size = 8
	temp_warp_size = pixelcount * temp_pixel_size
	
	if version > 1: 
        # checking if the QR code has a version greater 1. Then it has a alignment
		# pattern at the bottom right corner, too
		# trying to find this alignment pattern for warping


		# The width and height of the area for searching the contour of the
		# finder pattern will be 2 * aligment_area_delta
		alignment_area_delta = pixel_lenght * 8 

		# calculating the middle of the qr code with the centers of the pattern
		# TR and BL
		# calculating the vector from pattern_corner_list[TL][BR] to the middle
		# of the qr code
		# adding this vector two times to pattern_corner_list[TL][BR] to reach
		# the pattern_corner_list[BR][TL] (which values do not exist yet)
		# this coordinates are a half pixel away from the center.  
		# the area of searching this center should be big enough to handle this offset
		estimated_alignment_center = (pattern_center_list[TR] + pattern_center_list[BL] - pattern_corner_list[TL][BR])

		# finding possible alignment pattern
		possible_alignment_index_list = (i for i in range(len(hierarchy)) if get_dept(i) == 3)

		# filtering this by only checking if one contour point is inside a
		# defined squared area close to the estimated center of the alignment
		# pattern
		possible_alignment_index_list_prefiltered = (i for i in possible_alignment_index_list if all(numpy.abs(contours[i][0][0] - estimated_alignment_center) < alignment_area_delta))
		
		# calculating the center of all possible finder pattern
		possible_alignment_moment_list = (cv2.moments(contours[i]) for i in possible_alignment_index_list_prefiltered)
		possible_alignment_centers = [numpy.array([moments['m10'], moments['m01']]) / moments['m00'] for moments in possible_alignment_moment_list]

	else: # version 1 contains no alignment patterns
		possible_alignment_centers = []

	# defining the warp source and destination quadrangle
	if possible_alignment_centers: # if we found the alignment pattern at bottom right, use this
		# if there are several possibilities for the alignment pattern, use the
		# closest to the estimation
		_, br_alignment_center = min((numpy.linalg.norm(center - estimated_alignment_center), center) for center in possible_alignment_centers)
		source = numpy.array([pattern_corner_list[TL][TL], pattern_corner_list[TR][TR], br_alignment_center, pattern_corner_list[BL][BL]], dtype = numpy.float32)
		destination = numpy.array([(0, 0), (temp_warp_size, 0), (temp_warp_size - temp_pixel_size * 6.5, temp_warp_size - temp_pixel_size * 6.5), (0, temp_warp_size)], dtype = numpy.float32)
	else: # otherwise use the extrapolated bottom right corner
		# extrapolation of the bottom right corner
        # http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
		# This must be very exact
		tr_r_dif_vertical = pattern_corner_list[TR][TR] - pattern_corner_list[TR][BR]
		bl_b_dif_horizontal = pattern_corner_list[BL][BL] - pattern_corner_list[BL][BR]   
		t = numpy.cross(pattern_corner_list[BL][BL] - pattern_corner_list[TR][TR], bl_b_dif_horizontal) / numpy.cross(tr_r_dif_vertical, bl_b_dif_horizontal)
		br_br = pattern_corner_list[TR][TR] + t * tr_r_dif_vertical

		source = numpy.array([pattern_corner_list[TL][TL], pattern_corner_list[TR][TR], br_br, pattern_corner_list[BL][BL]], dtype = numpy.float32)
		destination = numpy.array([(0, 0), (temp_warp_size, 0), (temp_warp_size, temp_warp_size), (0, temp_warp_size)], dtype = numpy.float32)

	# performing the warping and thresholding
	warp_matrix = cv2.getPerspectiveTransform(source, destination)
	bigqr_nottresholded = cv2.warpPerspective(image_gray, warp_matrix, (temp_warp_size, temp_warp_size))	
	bigqr = cv2.adaptiveThreshold(bigqr_nottresholded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, temp_pixel_size * 10 + 1, 2)

	# resizing to the real amount of pixels and thresholding again
	qr_notresholded = cv2.resize(bigqr[0 : temp_warp_size, 0 : temp_warp_size], (pixelcount, pixelcount))
	a ,qr = cv2.threshold(qr_notresholded, 127, 255, cv2.THRESH_BINARY)
	
	# extracting the data
	inverted_data = numpy.asarray(qr, dtype = numpy.bool)

	if output_size:
		cv2.imshow('1 resized',cv2.resize(image_resized, (output_size,output_size)))
		cv2.imshow('2 gray',cv2.resize(image_gray , (output_size,output_size)))
		cv2.imshow('3 canny', cv2.resize(edges, (output_size,output_size)))
		if version > 1: 
			# getting the start and the end of the slice for each coordinate of
			# the alignment center
			# (numpy style adding a 2 element column vector to a 2 element row
			# vector will create a matrix with 4 elements)
			slice_coordinates = (numpy.transpose([estimated_alignment_center]) + [-alignment_area_delta, alignment_area_delta]).astype(int) 
			slice_coordinates[:,0] = numpy.maximum(slice_coordinates[:,0], [0,0])
			slice_coordinates[:,1] = numpy.minimum(slice_coordinates[:,1], edges.shape)
			area = (slice(*slice_coordinates[1]), slice(*slice_coordinates[0])) # get slice objects for the search area
			cv2.imshow('alignment pattern searching area', edges[area]) 

		cv2.imshow('4 big qr nottresholded', bigqr_nottresholded)
		cv2.imshow('5 big qr', bigqr)
		cv2.imshow('6 qr small', qr,)
		cv2.imshow('7 qr', cv2.resize(qr, (temp_warp_size, temp_warp_size), interpolation = cv2.INTER_NEAREST))
	return numpy.logical_not(inverted_data)
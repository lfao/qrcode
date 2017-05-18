from __future__ import division
from future.utils import raise_from
from builtins import zip
from builtins import chr
from builtins import str
from builtins import range

import numpy
import itertools
import reedsolo

class QrDecoderError(Exception):
	pass 

def extract_int_list(numpy_array, start_bit, word_length, word_count):
	'''
	Extracts a numpy array of integers out of a numpy array of booleans by interpreting the booleans as bits.
	Keyword arguments:
	numpy_array -- the numpy array of booleans
	start_bit -- the index of the first bit to be extracted
	word_length -- the amount of booleans being interpreted as bits for each integer
	word_count -- the amount of integers being extracted
	Returns:
	A numpy array of integers with the length word_count
	'''
	mydata = numpy_array[start_bit:(start_bit + word_count * word_length)].reshape(word_count, word_length)
	weight = 2 ** numpy.arange(word_length - 1, -1,  -1)
	return numpy.sum(weight * mydata, 1)

def extract_int(numpy_array, start_bit, word_length):
	'''
	Extracts a integer out of a numpy array of booleans by interpreting the booleans as bits.
	Keyword arguments:
	numpy_array -- the numpy array of booleans
	start_bit -- the index of the first bit to be extracted
	word_length -- the amount of booleans being interpreted as bits for the integer
	Returns:
	A integer with the length word_count
	'''
	weight = 2 ** numpy.arange(word_length - 1, -1,  -1)
	mydata = numpy_array[start_bit:(start_bit + word_length)]    
	return numpy.sum(weight * mydata)

def extract_long(array):
	'''
	Extracts a long integer out of a numpy array of booleans by interpreting the booleans as bits.
	Keyword arguments:
	numpy_array -- the numpy array of booleans
	Returns:
	A long integer
	'''
	return sum(1 << i for i, value in enumerate(array[::-1]) if value)

def get_dataarea_indicator(version):
	'''
	Generates a numpy matrix of booleans which values indicate, whether the field of a qr code with the same index contains data or a function pattern
	Keyword arguments:
	version -- the version of the qr code
	Returns:
	A numpy matrix of booleans which values are true if the field of a qr code with the same index contains data
	'''
	size = version * 4 + 17
	retval = numpy.ones((size,size), dtype=bool)    # generating a matrix of true booleans which are indicating the data area


	# removing timing pattern
	retval[6, :] = False
	retval[:, 6] = False
	
	# removing finder pattern
	retval[:9, :9] = False
	retval[size - 8 :, : 9] = False
	retval[: 9, size - 8 :] = False            
	
	if version > 1 : # removing alignment pattern for every version containing these patterns
		alignment_gaps_count = version // 7 + 1
		alignment_start = 6
		alignment_end = size - 7
		alignment_distance = int(((alignment_end - alignment_start) / alignment_gaps_count + 1.5) / 2) * 2
		alignment_start_remaining = alignment_end - (alignment_gaps_count - 1) * alignment_distance
		
		# calculating the coordinates of the center points of all finder pattern
		# the first row and the first column contain a different amount of
		# alignment pattern because of the finder pattern
		# the distances to the remaining alignment patterns are
		# different for some versions
		alignment_position_generator_first_col = ((alignment_start, alignment_start_remaining + col_factor * alignment_distance) for col_factor in range(alignment_gaps_count - 1))
		alignment_position_generator_first_row = ((alignment_start_remaining + row_factor * alignment_distance, alignment_start) for row_factor in range(alignment_gaps_count - 1))
		
		# the alignment patterns always have the same gaps so they are created
		# in a nested loop (itertools.product)
		alignment_position_generator_remaining = ((alignment_start_remaining + row_factor * alignment_distance, 
													alignment_start_remaining + col_factor * alignment_distance) 
													for row_factor, col_factor in itertools.product(range(alignment_gaps_count), repeat = 2))

		# chaining all generators of alignment patterns to one
		alignment_position_generator = itertools.chain(alignment_position_generator_first_col, alignment_position_generator_first_row, alignment_position_generator_remaining)
		
		# removing the area for each finder pattern
		for row, col in alignment_position_generator:
			retval[row - 2: row + 3, col - 2: col + 3] = False

	if version >= 7: # removing version info fields for every version containing these fields
		retval[size - 11 : size - 8, :6] = False
		retval[:6, size - 11 : size - 8] = False

	return retval

def get_version_size(bit_matrix):
	'''
	Calculates the version of the qr code and the length of a side of the squared associated data matrix
	Keyword arguments:
	bit_matrix -- the boolean data matrix of the qr code
	Returns:
	A tuple containing the version of the qr code (1-40) and the length of a side of the squared associated data matrix
	'''
	size, _ = bit_matrix.shape
	version = (size - 17) // 4
	return version, size


def get_version_info(bit_matrix):
	'''
	Extracts the version block of a data matrix
	Keyword arguments:
	bit_matrix -- the boolean data matrix of the qr code
	Returns:
	A numpy array of booleans with 18 fields containing the version information
	'''
	version, size = get_version_size(bit_matrix)
	return numpy.transpose(bit_matrix[size - 9 : size - 12: -1, 5::-1]).flatten() if version >= 7 else None

def get_format_info_data(bit_matrix):
	'''
	Extracts the format info block of a data matrix
	Keyword arguments:
	bit_matrix -- the boolean data matrix of the qr code
	Returns:
	A tuple of the index of the applied mask (0-7) and the level of error correction (0-3)
	'''
	format_info = numpy.append(bit_matrix[8,[0, 1, 2, 3, 4, 5, 7,8]], bit_matrix[[7, 5, 4, 3, 2, 1, 0], 8])
	format_info = numpy.logical_xor([True, False, True, False, True, False, False, False, False, False, True, False, False, True, False], format_info, format_info)
	ecc_level = extract_int(format_info, 0, 2) ^ 1
	mask = extract_int(format_info, 2, 3)
	
	return mask, ecc_level


def extract_bit_array(bit_matrix, mask_index, output=False):
	'''
	Extracts the data block of a data matrix
	Keyword arguments:
	bit_matrix -- the data matrix of the qr code
	mask_index -- the index of the applied mask (0-7)
	Returns:
	A list of booleans being extracted from the data matrix
	'''

	# the list of mask functions (0-7) being used for creating the mask matrix
	MASK_FUNCTIONS = [lambda row, column : (row + column) % 2 == 0, 
		lambda row, column : (row) % 2 == 0, 
		lambda row, column : (column) % 3 == 0, 
		lambda row, column : (row + column) % 3 == 0, 
		lambda row, column : ((row // 2) + (column // 3)) % 2 == 0, 
		lambda row, column : ((row * column) % 2) + ((row * column) % 3) == 0, 
		lambda row, column : (((row * column) % 2) + ((row * column) % 3)) % 2 == 0, 
		lambda row, column : (((row + column) % 2) + ((row * column) % 3)) % 2 == 0]
	
	version, size = get_version_size(bit_matrix)

	dataarea_indicator = get_dataarea_indicator(version)  
	mask_matrix = numpy.fromfunction(MASK_FUNCTIONS[mask_index], bit_matrix.shape, dtype = int) # creating the raw mask_matrix
	mask_matrix = numpy.logical_and(mask_matrix, dataarea_indicator, mask_matrix)  # removing the parts which contain no data
	bit_matrix = numpy.logical_xor(bit_matrix, mask_matrix, bit_matrix) # inverting the pixels of the original image, which are indicated by the mask
	
	# creating lists for going up and down a data row consisting of two pixel
	# rows
	# this contains tuples of the horizontal offset 0 or 1 (column offset) and
	# the vertical position (row)
	index_column_gen_up = [(i % 2, size - 1 - i // 2) for i in range(2 * size)]
	index_column_gen_down = [(i % 2,            i // 2) for i in range(2 * size)]  
	
	# creating generators which can be used for alternating going up and down
	# this generates tuples of horizontal positions without the offset and the
	# generators of going up or down a column
	# the amount of tuples only depends on the range because itertools.cycle
	# repeats its elements infinitely
	# later the offset must be subtracted from the horizontal position to get
	# the real horizontal position
	# the vertical timing pattern is a irregularity for data extraction
	# Therefor generators for each side of the timing pattern are created
	# after creating both, the two generators are chained to one
	offset_indexlist_tuple_gen_right = zip(range(size - 1, 7, -2), itertools.cycle([index_column_gen_up, index_column_gen_down]))
	offset_indexlist_tuple_gen_left = zip(range(5,        0, -2), itertools.cycle([index_column_gen_down, index_column_gen_up]))
	offset_indexlist_tuple_gen = itertools.chain(offset_indexlist_tuple_gen_right, offset_indexlist_tuple_gen_left) 

	# the generated tuples of columns without offset and the lists of offsets
	# and the row must be combined
	# therefor the offset is subtracted from the column
	indexlist_unfiltered = ((row, col - delta) for col, column_gen in offset_indexlist_tuple_gen for delta, row in column_gen)

	# the tuples would contain all fields, even if it is no data field but for
	# example finder pattern, alignment pattern or other
	# these have to be filtered out. Therefor the dataarea_indicator is used.
	# It is true if the matrix contains data at the specified coordinates
	indexlist = ((row, col) for (row, col) in indexlist_unfiltered if dataarea_indicator[row, col])

	# Finally bits are extracted from the bit matrix in the right order
	# the generator of tuples of coordinates is converted to a tuple of lists
	# with zip
	raw_bit_array = bit_matrix[tuple(zip(*indexlist))]

	if output:
		return raw_bit_array, (mask_matrix, dataarea_indicator, bit_matrix)
	else:
		return raw_bit_array

def error_correction(raw_bit_array, version, ecc_level):
	'''
	Extracts the error-corrected data out of a raw bit array
	Keyword arguments:
	raw_bit_array -- the list of booleans being extracted from the data matrix
	version -- the version of the qr code (1-40)
	ecc_level -- the level of the error correction (0-3)
	Returns:
	A list of booleans, which have been corrected with the error-correction algorithm (This will contain less elements than the raw_bit_array)
	'''

	CODEWORD_COUNT_LOOKUP = [[19, 16, 13, 9], [34, 28, 22, 16], [55, 44, 34, 26], [80, 64, 48, 36], [108, 86, 62, 46], 
							[136, 108, 76, 60], [156, 124, 88, 66], [194, 154, 110, 86], [232, 182, 132, 100], [274, 216, 154, 122],
							[324, 254, 180, 140], [370, 290, 206, 158], [428, 334, 244, 180], [461, 365, 261, 197], [523, 415, 295, 223], 
							[589, 453, 325, 253], [647, 507, 367, 283], [721, 563, 397, 313], [795, 627, 445, 341], [861, 669, 485, 385], 
							[932, 714, 512, 406], [1006, 782, 568, 442], [1094, 860, 614, 464], [1174, 914, 664, 514], [1276, 1000, 718, 538], 
							[1370, 1062, 754, 596], [1468, 1128, 808, 628], [1531, 1193, 871, 661], [1631, 1267, 911, 701], [1735, 1373, 985, 745], 
							[1843, 1455, 1033, 793], [1955, 1541, 1115, 845], [2071, 1631, 1171, 901], [2191, 1725, 1231, 961], [2306, 1812, 1286, 986], 
							[2434, 1914, 1354, 1054], [2566, 1992, 1426, 1096], [2702, 2102, 1502, 1142], [2812, 2216, 1582, 1222], [2956, 2334, 1666, 1276]]
	BLOCK_COUNT_LOOKUP = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 2, 2], [1, 2, 2, 4], [1, 2, 4, 4], 
						  [2, 4, 4, 4], [2, 4, 6, 5], [2, 4, 6, 6], [2, 5, 8, 8], [4, 5, 8, 8], 
						  [4, 5, 8, 11], [4, 8, 10, 11], [4, 9, 12, 16], [4, 9, 16, 16], [6, 10, 12, 18], 
						  [6, 10, 17, 16], [6, 11, 16, 19], [6, 13, 18, 21], [7, 14, 21, 25], [8, 16, 20, 25], 
						  [8, 17, 23, 25], [9, 17, 23, 34], [9, 18, 25, 30], [10, 20, 27, 32], [12, 21, 29, 35], 
						  [12, 23, 34, 37], [12, 25, 34, 40], [13, 26, 35, 42], [14, 28, 38, 45], [15, 29, 40, 48], 
						  [16, 31, 43, 51], [17, 33, 45, 54], [18, 35, 48, 57], [19, 37, 51, 60], [19, 38, 53, 63], 
						  [20, 40, 56, 66], [21, 43, 59, 70], [22, 45, 62, 74], [24, 47, 65, 77], [25, 49, 68, 81]]
	
	block_count = BLOCK_COUNT_LOOKUP[version - 1][ecc_level] # the amount of blocks being interwoven
	codeword_count = CODEWORD_COUNT_LOOKUP[version - 1][ecc_level] # the sum of data bytes without error-correction overhead being contained in
																   # all blocks
	
	bytes_count = raw_bit_array.size // 8 # the amount of entire bytes contained in the bit array
	# the amount of data bytes without error-correction overhead contained in a short block
	short_codeword_block_bytes_count = codeword_count // block_count 
	#long_codeword_block_bytes_count = short_block_bytes_count + 1 
	# not required. The amount of bytes in long blocks is 1 greater than the short blocks

	# the amount of bytes in the error-correction part of each block
	errorcorrection_block_bytes_count = (bytes_count - codeword_count) // block_count 

	long_codeword_block_count = codeword_count % block_count    # The amount of long data blocks
	short_codeword_block_count = block_count - long_codeword_block_count # The amount of short data blocks
	 

	# generators of indices for the data block.
	# The indices of the short blocks are completely regular and always have the
	# distance block_count
	# The indices of the long blocks are regularly exceptional the last index.
	# The last index only has the distance long_codeword_block_count instead of
	# block_count from the next to the last
	# Therefor two different generators are generated and chained
	short_codeword_block_index_list_gen = (range(block, short_codeword_block_bytes_count * block_count, block_count) 
											for block in range(short_codeword_block_count))
	long_codeword_block_index_list_gen = (itertools.chain(range(block, short_codeword_block_bytes_count * block_count, block_count), 
											 [short_codeword_block_bytes_count * block_count + block]) 
											for block in range(short_codeword_block_count, block_count))
	codeword_block_index_list_gen = itertools.chain(short_codeword_block_index_list_gen, long_codeword_block_index_list_gen)
	
	# The error-correction blocks are all of the same size, have a regular
	# distance and start after all the data blocks
	errorcorrection_block_index_list_gen = (range(codeword_count + block, bytes_count, block_count) for block in range(block_count))

	# extracting the data and the correction data for each block
	codeword_errorcorrection_block_index_list_gen = (list(itertools.chain(block, correction_data)) for block, correction_data in zip(codeword_block_index_list_gen, errorcorrection_block_index_list_gen))

	raw_byte_array = numpy.packbits(raw_bit_array)  # converting the array of bits to arrays of bytes

	# applying reed Solomon error-correction to all blocks
	# the result is a generator of numpy arrays of corrected bytes
	corrected_byte_list_gen = (reedsolo.rs_correct_msg(raw_byte_array[codeword_errorcorrection_block_index_list], errorcorrection_block_bytes_count) 
								for codeword_errorcorrection_block_index_list in codeword_errorcorrection_block_index_list_gen)

	# concatenating the bytes of each block to one array
	corrected_byte_array = numpy.fromiter(itertools.chain.from_iterable(corrected_byte_list_gen), dtype = numpy.uint8, count = codeword_count)
	
	# extracting the bits from the bytes
	corrected_bit_array = numpy.unpackbits(corrected_byte_array)

	return corrected_bit_array

def extract_string(corrected_bit_array, version):
	'''
	Extracts the string out of the error-corrected data
	Keyword arguments:
	corrected_bit_array -- the list of booleans being corrected by the error-correction algorithm
	version -- the version of the qr code (1-40)
	Returns:
	The string contained in the qr code
	'''
	SUPPORTED_MODES = [1,2,4]
	LENGTH_CODE_LENGTH_LOOKUP = [[10, 9, 8, 8],[12, 11, 16, 10],[14, 13, 16, 12]]
	WORD_LENGTH_LOOKUP = [10,11,8]
	CHARS_PER_WORD_LOOKUP = [3,2,1,1]
	version_index = 0 if version < 10 else 1 if version < 27 else 2    

	mode_value = extract_int(corrected_bit_array, 0, 4)
	next_block_start = 0
	result_string = ''
	try:
		while mode_value in SUPPORTED_MODES: #TODO: testing every mode, mixed modes
			mode_index = SUPPORTED_MODES.index(mode_value)
		
			# the amount of bits of the length code
			length_code_length = LENGTH_CODE_LENGTH_LOOKUP[version_index][mode_index]

			# the amount of bits of a data word
			word_length = WORD_LENGTH_LOOKUP[mode_index]

			# the amount of characters coded inside the qr code
			char_count = extract_int(corrected_bit_array, 4 + next_block_start, length_code_length)

			# the start point of the data
			data_beginn = 4 + length_code_length + next_block_start
		
			# the amount of complete words
			word_count = char_count // CHARS_PER_WORD_LOOKUP[mode_index]
			# and the amount of characters in the incomplete last word.
			# zero, if it does not exist
			remaining_chars_count = char_count % CHARS_PER_WORD_LOOKUP[mode_index]

			# the part of the data which is coded in complete words
			# Maybe there are some characters left which are coded in less than
			# word_length bits
			int_list = extract_int_list(corrected_bit_array, data_beginn, word_length, word_count)

			# the position of the remaining characters in the bit array
			remaining_chars_start = data_beginn + word_length * word_count

			if mode_index == 0: # numeric mode - words of 10 bits contain 3 chars (0 - 9)
				# extracting the complete words of 3 chars
				result_string += ''.join('{:03d}'.format(item) for item in int_list)

				if remaining_chars_count > 0:
					# extracting 1 or 2 remaining characters in the incomplete word
					# of 4 or 8 bits
					remaining_ints = extract_int(corrected_bit_array, remaining_chars_start, remaining_chars_count * 4)
					# appending the remaining characters to the result
					result_string += '{0:0{1}d}'.format(remaining_ints, remaining_chars_count)

				next_block_start = remaining_chars_start + remaining_chars_count * 4
			elif mode_index == 1: # alphanumeric mode - words of 11 bits contain 2 chars (0 - 9, A - Z, $, %, *,
								  # +, -, ., /, :)
				CHAR_LOOKUP = [str(i) for i in range(10)] + [chr(65 + i) for i in range(26)] + list(' $%*+-./:')

				# extracting the complete words of 2 characters
				result_string +=  ''.join(CHAR_LOOKUP[item // 45] + CHAR_LOOKUP[item % 45] for item in int_list)

				if remaining_chars_count > 0:
					# extracting 1 remaining character in the incomplete word of 6
					# bits and appending it to the result
					result_string += CHAR_LOOKUP[extract_int(corrected_bit_array, remaining_chars_start, 6)]
				next_block_start = remaining_chars_start + remaining_chars_count * 6

			elif mode_index == 2: # byte mode - words of 8 bits contain 1 char (ASCII)
				# extracting words of 1 character  
				# Since there is only 1 character per word, no incomplete words exist
				result_string +=  ''.join(chr(item) for item in int_list)
				next_block_start = remaining_chars_start

			# mode value is 0 if all blocks end here
			# maybe further data with another mode follow here
			mode_value = extract_int(corrected_bit_array, next_block_start, 4)
	except(ValueError, IndexError) as e:
		#raise QrDecoderError('QR data inconsistent!') from e
		raise_from(QrDecoderError('QR data inconsistent!'), e)

	return result_string

def decode(bit_matrix):
	'''
	Extracts the string out of a qr code matrix containing a boolean for every pixel
	Keyword arguments:
	bit_matrix -- a qr code matrix containing a boolean for every pixel
	Returns:
	The string contained in the qr code
	'''
	mask_index, ecc_level = get_format_info_data(bit_matrix)
	version, size = get_version_size(bit_matrix)

	bit_array_raw = extract_bit_array(bit_matrix, mask_index)
	bit_array = error_correction(bit_array_raw, version, ecc_level)
	string = extract_string(bit_array, version)
	return string
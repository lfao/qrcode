import numpy
import itertools
import reedsolo

def extract_int_list(numpy_array, start_bit, word_length, word_count):
    '''
    Extract a numpy array of intergers out of a numpy array of boolean by interpreting the boolean as bit.
    Keyword arguments:
    numpy_array -- the numpy array of boolean
    start_bit -- the index of the first bit to be extracted
    word_length -- the amount of bools being interpreted as bits for each integer
    word_count -- the amount of integer being extracted
    Returns:
    A numpy array of integer with the length word_count
    '''
    mydata = numpy_array[start_bit:(start_bit + word_count * word_length)].reshape(word_count, word_length)
    weight = 2 ** numpy.arange(word_length - 1, -1,  -1)
    return numpy.sum(weight * mydata, 1)

def extract_int(numpy_array, start_bit, word_length):
    '''
    Extract a interger out of a numpy array of boolean by interpreting the boolean as bit.
    Keyword arguments:
    numpy_array -- the numpy array of boolean
    start_bit -- the index of the first bit to be extracted
    word_length -- the amount of bools being interpreted as bits for the integer
    Returns:
    A integer with the length word_count
    '''
    weight = 2 ** numpy.arange(word_length - 1, -1,  -1)
    mydata = numpy_array[start_bit:(start_bit + word_length)]
    return numpy.sum(weight * mydata)

def extract_long(array):
    '''
    Extract a long integer out of a numpy array of boolean by interpreting the boolean as bit.
    Keyword arguments:
    numpy_array -- the numpy array of boolean
    Returns:
    A long integer
    '''
    return sum(1L << i for i, value in enumerate(array[::-1]) if value)

def get_dataarea_indicator(version):
    '''
    Generate a numpy matrix of bools which values indicate, wether the field of an qr code with the same index contains data or functionpattern
    Keyword arguments:
    version -- the version of the qr code
    Returns:
    A numpy matrix of boolean which values are true if the field of an qr code with the same index contains data
    '''
    size = version * 4 + 17
    retval = numpy.ones((size,size), dtype=bool)
    retval[6, :] = False
    retval[:, 6] = False
    retval[:9, :9] = False
    retval[size - 8 :, : 9] = False
    retval[: 9, size - 8 :] = False            
    
    if version > 1 :
        alignment_gaps_count = version / 7 + 1
        alignment_start = 6
        alignment_end = size - 7
        alignment_distance = int((float(alignment_end - alignment_start) / alignment_gaps_count + 1.5) / 2) * 2
        alignment_start_remaining = alignment_end - (alignment_gaps_count - 1) * alignment_distance
        #print alignment_distance, alignment_start_remaining, "alignment right left"
            
        alignment_position_generator_first_col = ((alignment_start, alignment_start_remaining + col_factor * alignment_distance) for col_factor in xrange(alignment_gaps_count - 1))
        alignment_position_generator_first_row = ((alignment_start_remaining + row_factor * alignment_distance, alignment_start) for row_factor in xrange(alignment_gaps_count - 1))
        alignment_position_generator_remaining = ((alignment_start_remaining + row_factor * alignment_distance, 
                                                    alignment_start_remaining + col_factor * alignment_distance) 
                                                    for row_factor, col_factor in itertools.product(xrange(alignment_gaps_count), repeat = 2))
        alignment_position_generator = itertools.chain(alignment_position_generator_first_col, alignment_position_generator_first_row, alignment_position_generator_remaining)
         
        for row, col in alignment_position_generator:
            retval[row - 2: row + 3, col - 2: col + 3] = False

    if version >= 7:
        retval[size - 11 : size - 8, :6] = False
        retval[:6, size - 11 : size - 8] = False

    return retval

def get_version_size(bit_matrix):
    '''
    Calculates the version of the qr code and the lenght of a side of the squared associated data matrix
    Keyword arguments:
    bit_matrix -- the boolean data matrix of the qr code
    Returns:
    A tuple containing the version of the qr code (1-40) and the lenght of a side of the squared associated data matrix
    '''
    size, _ = bit_matrix.shape
    version = (size - 17) / 4
    return version, size


def get_version_info(bit_matrix):
    '''
    Extracts the version block of a data matrix
    Keyword arguments:
    bit_matrix -- the boolean data matrix of the qr code
    Returns:
    A numpy array of boolean with 18 fields containing the version information
    '''
    version, size = get_version_size(bit_matrix)
    return numpy.transpose(bit_matrix[size - 9 : size - 12: -1, 5::-1]).flatten() if version >= 7 else None

def get_format_info_data(bit_matrix):
    '''
    Extracts the format info block of a data matrix
    Keyword arguments:
    bit_matrix -- the boolean data matrix of the qr code
    Returns:
    A tupel of the index of the applied mask (0-7) and the level of error correction(0-3)
    '''
    format_info = numpy.append(bit_matrix[8,[0, 1, 2, 3, 4, 5, 7,8]], bit_matrix[[7, 5, 4, 3, 2, 1, 0], 8])
    format_info = numpy.logical_xor([True, False, True, False, True, False, False, False, False, False, True, False, False, True, False], format_info, format_info)
    ecc_level = extract_int(format_info, 0, 2) ^ 1
    mask = extract_int(format_info, 2, 3)
    
    return mask, ecc_level


def extract_bit_array(bit_matrix, mask_index):
    '''
    Extracts the data block of a data matrix
    Keyword arguments:
    bit_matrix -- the data matrix of the qr code
    mask_index -- the index of the applied mask (0-7)
    Returns:
    A list of booleans being extracted from the data matrix
    '''
    MASK_FUNCTIONS = [
        lambda row, column : (row + column) % 2 == 0, 
        lambda row, column : (row) % 2 == 0, 
        lambda row, column : (column) % 3 == 0, 
        lambda row, column : (row + column) % 3 == 0, 
        lambda row, column : ( numpy.floor(row / 2) + numpy.floor(column / 3) ) % 2 == 0, 
        lambda row, column : ((row * column) % 2) + ((row * column) % 3) == 0, 
        lambda row, column : ( ((row * column) % 2) + ((row * column) % 3) ) % 2 == 0, 
        lambda row, column : ( ((row + column) % 2) + ((row * column) % 3) ) % 2 == 0
        ]
    
    version, size = get_version_size(bit_matrix)

    dataarea_indicator = get_dataarea_indicator(version)
    mask_matrix = numpy.fromfunction(MASK_FUNCTIONS[mask_index], bit_matrix.shape)
    mask_matrix = numpy.logical_and(mask_matrix, dataarea_indicator, mask_matrix)   
    bit_matrix = numpy.logical_xor(bit_matrix, mask_matrix, bit_matrix)
    
    if False:
        import cv2
        cv2.imshow("mask", cv2.resize(numpy.logical_not(mask_matrix).astype(float), (size * 8, size * 8), interpolation = cv2.INTER_NEAREST))
        cv2.imshow("ausgeblendet", cv2.resize(dataarea_indicator.astype(float), (size * 8, size * 8), interpolation = cv2.INTER_NEAREST))
        cv2.imshow("data", cv2.resize(numpy.logical_not(bit_matrix).astype(float), (size * 8, size * 8), interpolation = cv2.INTER_NEAREST))
    
    index_column_gen_up   = [(i % 2, size - 1 - i / 2) for i in xrange(2 * size)]
    index_column_gen_down = [(i % 2,            i / 2) for i in xrange(2 * size)]    
    offset_indexlist_tuple_gen_right = itertools.izip(xrange(size - 1, 7, -2), itertools.cycle([index_column_gen_up, index_column_gen_down]))
    offset_indexlist_tuple_gen_left  = itertools.izip(xrange(5,        0, -2), itertools.cycle([index_column_gen_down, index_column_gen_up]))
    offset_indexlist_tuple_gen       = itertools.chain(offset_indexlist_tuple_gen_right, offset_indexlist_tuple_gen_left) 
    indexlist_unfiltered = ((row, col - delta) for col, column_gen in offset_indexlist_tuple_gen for delta, row in column_gen)
    indexlist = ((row, col) for (row, col) in indexlist_unfiltered if dataarea_indicator[row, col])
    raw_bit_array = bit_matrix[zip(*indexlist)]
   
    return raw_bit_array

def error_correction(raw_bit_array, version, ecc_level):
    '''
    Extracts the error corrected data out of a raw bit array
    Keyword arguments:
    raw_bit_array -- the list of booleans being extracted from the data matrix
    version -- the version of the qr code (1-40)
    ecc_level -- the level of the error correction (0-3)
    Returns:
    A list of booleans, which have been corrected whith the error correction algorithm (This will contian less elements than raw_bit_array)
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
    
    block_count    = BLOCK_COUNT_LOOKUP   [version - 1][ecc_level]
    codeword_count = CODEWORD_COUNT_LOOKUP[version - 1][ecc_level]
    
    bytes_count = raw_bit_array.size / 8
    short_codeword_block_bytes_count = codeword_count / block_count
    #long_codeword_block_bytes_count = short_block_bytes_count + 1 # not required

    long_codeword_block_count = codeword_count % block_count
    # short_codeword_block_count = block_count - long_block_count # not required
    errorcorrection_block_bytes_count = (bytes_count - codeword_count) / block_count

    short_codeword_block_index_list_gen = (range(block, short_codeword_block_bytes_count * block_count, block_count) 
                                            for block in xrange(block_count - long_codeword_block_count))
    long_codeword_block_index_list_gen  = (range(block, short_codeword_block_bytes_count * block_count, block_count) 
                                            + [short_codeword_block_bytes_count * block_count + block] 
                                            for block in xrange(block_count - long_codeword_block_count, block_count))
    codeword_block_index_list_gen = itertools.chain(short_codeword_block_index_list_gen, long_codeword_block_index_list_gen)
        
    errorcorrection_block_index_list_gen = (range(block + codeword_count, bytes_count, block_count) for block in xrange(block_count))

    raw_byte_array = numpy.packbits(raw_bit_array)
    corrected_byte_list_gen = (reedsolo.rs_correct_msg(raw_byte_array[block + correction_data], errorcorrection_block_bytes_count)
                       for block, correction_data in itertools.izip(codeword_block_index_list_gen, errorcorrection_block_index_list_gen))
    corrected_byte_array = numpy.fromiter(itertools.chain.from_iterable(corrected_byte_list_gen), dtype = numpy.uint8)
    corrected_bit_array = numpy.unpackbits(corrected_byte_array)

    return corrected_bit_array

def extract_string(corrected_bit_array, version):
    '''
    Extracts the error corrected data out of a raw bit array
    Keyword arguments:
    corrected_bit_array -- the list of booleans being corrected by the error correction algorithm
    version -- the version of the qr code (1-40)
    Returns:
    A string being contained in the qr code
    '''

    version_index = 0 if version < 10 else 1 if version < 27 else 2    

    mode_value = extract_int(corrected_bit_array, 0, 4)
    next_block_start = 0
    result_string = ""

    while mode_value in [1,2,4]: #TODO: testing every mode, mixed modes
        mode_index = int(numpy.log2(mode_value))
        length_code_length = [[10, 9, 8, 8],[12, 11, 16, 10],[14, 13, 16, 12]][version_index][mode_index]
        word_length = [10,11,8][mode_index]
        char_count =  extract_int(corrected_bit_array, 4 + next_block_start, length_code_length)
        data_beginn = 4 + length_code_length + next_block_start
        word_count = char_count / [3,2,1,1][mode_index]
        chars_remaining = char_count % word_count
        int_list = extract_int_list(corrected_bit_array, data_beginn, word_length, word_count)

        remaining_start = data_beginn + word_length * word_count

        if mode_index == 0:
            result_string += "".join("{:03d}".format(item) for item in int_list)
            remaining_ints = extract_int(corrected_bit_array, remaining_start, chars_remaining * 4)
            next_block_start = remaining_start + chars_remaining * 4
            result_string += "{0:0{1}d}".format(remaining_ints, chars_remaining)
        elif mode_index == 1: #TODO: testing
            CHAR_LOOKUP = [str(i) for i in range(10)] + [chr(65 + i) for i in range(26)] + list(' $%*+-./:')

            result_string +=  "".join(CHAR_LOOKUP[item / 45] + CHAR_LOOKUP[item % 45] for item in int_list)
            if chars_remaining:
                result_string += CHAR_LOOKUP[extract_int(corrected_bit_array, remaining_start, 6)]
            next_block_start = remaining_start + chars_remaining * 6
        elif mode_index == 2:
            result_string +=  "".join(chr(item) for item in int_list)
            next_block_start = remaining_start

        mode_value = extract_int(corrected_bit_array, next_block_start,  4)

    return result_string


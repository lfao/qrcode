import numpy
import itertools
import operator
import functools

def extract_int_list(numpy_array, start_bit, word_len, word_count):
    mydata = numpy_array[start_bit:(start_bit + word_count * word_len)].reshape(word_count, word_len)
    weight = 2 ** numpy.arange(word_len - 1, -1 , -1)
    return numpy.sum(weight * mydata, 1)

def extract_int(numpy_array, start_bit, word_len):
    [retval] = extract_int_list(numpy_array, start_bit, word_len, 1)
    return retval

def extract_stream(data):
    MASK_FUNCTIONS = [
        lambda row, column : (row + column) % 2 == 0 ,
        lambda row, column : (row) % 2 == 0 ,
        lambda row, column : (column) % 3 == 0 ,
        lambda row, column : (row + column) % 3 == 0 ,
        lambda row, column : ( numpy.floor(row / 2) + numpy.floor(column / 3) ) % 2 == 0 ,
        lambda row, column : ((row * column) % 2) + ((row * column) % 3) == 0 ,
        lambda row, column : ( ((row * column) % 2) + ((row * column) % 3) ) % 2 == 0 ,
        lambda row, column : ( ((row + column) % 2) + ((row * column) % 3) ) % 2 == 0
        ]

    def get_dataarea_indicator(version):
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
            print alignment_distance, alignment_start_remaining, "alignment right left"
            
            alignment_position_generator_first_col = ((alignment_start, alignment_start_remaining + col_factor * alignment_distance) for col_factor in xrange(alignment_gaps_count - 1))
            alignment_position_generator_first_row = ((alignment_start_remaining + row_factor * alignment_distance, alignment_start) for row_factor in xrange(alignment_gaps_count - 1))
            alignment_position_generator_remaining = ((alignment_start_remaining + row_factor * alignment_distance, 
                                                       alignment_start_remaining + col_factor * alignment_distance) 
                                                       for row_factor, col_factor in itertools.product(xrange(alignment_gaps_count), repeat = 2))
            alignment_position_generator = itertools.chain(alignment_position_generator_first_col, alignment_position_generator_first_row, alignment_position_generator_remaining)

            for row, col in alignment_position_generator:
                retval[col - 2: col + 3, row - 2: row + 3] = False

        if version >= 7:
            retval[size - 11 : size - 8, :6] = False
            retval[:6, size - 11 : size - 8] = False

        return retval
    
    size, _ = data.shape
    version = (size - 17) / 4
    format_info = numpy.append(data[8,[0, 1, 2, 3, 4, 5, 7,8]], data[[7, 5, 4, 3, 2, 1, 0], 8])
    format_info = numpy.logical_xor([True, False, True, False, True, False, False, False, False, False, True, False, False, True, False], format_info, format_info)
    mask = extract_int(format_info, 2, 3)
    edc_level = extract_int(format_info, 0, 2)
    #print size, version
    print mask , edc_level , version, "mask, edc level, version"
    #print format_info   

    dataarea_indicator = get_dataarea_indicator(version)
    mask_matrix = numpy.fromfunction(MASK_FUNCTIONS[mask], data.shape)
    mask_matrix = numpy.logical_and(mask_matrix, dataarea_indicator, mask_matrix)   
    data = numpy.logical_xor(data, mask_matrix, data)
    
    if True:
        import cv2
        cv2.imshow("mask", cv2.resize(numpy.logical_not(mask_matrix).astype(float), (size * 8, size * 8), interpolation = cv2.INTER_NEAREST))
        cv2.imshow("ausgeblendet", cv2.resize(dataarea_indicator.astype(float), (size * 8, size * 8), interpolation = cv2.INTER_NEAREST))
        cv2.imshow("data", cv2.resize(numpy.logical_not(data).astype(float), (size * 8, size * 8), interpolation = cv2.INTER_NEAREST))
    
    index_upgen   = [(i % 2, size - 1 - i / 2) for i in xrange(2 * size)]
    index_downgen = [(i % 2,            i / 2) for i in xrange(2 * size)]    
    index_gens_right = itertools.izip(xrange(size - 1, 7, -2), itertools.cycle([index_upgen, index_downgen]))
    index_gens_left  = itertools.izip(xrange(5,        0, -2), itertools.cycle([index_downgen, index_upgen]))
    index_gens       = itertools.chain(index_gens_right, index_gens_left) 
    indices_unfiltered = ((row, col - delta) for col, gen in index_gens for delta, row in gen)
    indices = ((row, col) for (row, col) in indices_unfiltered if dataarea_indicator[row, col])
    values = data[zip(*indices)]
    
    version_info = None
    return values, version, format_info, version_info

def extract_data(values, version, format_info, version_info):
    mode_value = extract_int(values,0,4)
    mode_index = int(numpy.log2(mode_value))
    version_index = 0 if version < 10 else 1 if version < 27 else 2
    length_code_length = [[10, 9, 8, 8],[12, 11, 16, 10],[14 ,13 ,16 ,12]][version_index][mode_index]
    word_length = [10,11,8,8][mode_index]

    length_code_length = 8
    length =  extract_int(values, 4, length_code_length)
    data_beginn = 4 + length_code_length
    print mode_index ,  "modeindex"
    print length_code_length, length
    temp_length = length / [3,2,1,1][mode_index]
    #temp_length = length
    int_list = extract_int_list(values, data_beginn, word_length, temp_length)
    
    #temp_int_list = extract_int_list(values, 0, 8, 194)

    #print ' '.join("{:02x}".format(value) for value in temp_int_list)
    #print '""""""""""""""""""""""""""""""""""""""'
    
    #temp_int_list2 = numpy.append(temp_int_list[::2],temp_int_list[1::2])
    #print values[:40]
    #print ' '.join("{:02x}".format(value) for value in temp_int_list2)

    return int_list, mode_index
    #return temp_int_list2, mode_index


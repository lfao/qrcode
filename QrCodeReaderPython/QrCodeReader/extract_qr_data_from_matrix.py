import numpy
import itertools
import operator
import functools
import reedsolo

def extract_int_list(numpy_array, start_bit, word_len, word_count):
    mydata = numpy_array[start_bit:(start_bit + word_count * word_len)].reshape(word_count, word_len)
    weight = 2 ** numpy.arange(word_len - 1, -1 , -1)
    return numpy.sum(weight * mydata, 1)

def extract_int(numpy_array, start_bit, word_len):
    [retval] = extract_int_list(numpy_array, start_bit, word_len, 1)
    return retval

def extract_long(array):
    return sum(2L << i if value else 0 for i, value in enumerate(array[::-1]))

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
        #print alignment_distance, alignment_start_remaining, "alignment right left"
            
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
    
    size, _ = data.shape
    version = (size - 17) / 4
    format_info = numpy.append(data[8,[0, 1, 2, 3, 4, 5, 7,8]], data[[7, 5, 4, 3, 2, 1, 0], 8])
    format_info = numpy.logical_xor([True, False, True, False, True, False, False, False, False, False, True, False, False, True, False], format_info, format_info)
    mask = extract_int(format_info, 2, 3)

    #print mask, version, "mask, version"
    #print format_info   

    dataarea_indicator = get_dataarea_indicator(version)
    mask_matrix = numpy.fromfunction(MASK_FUNCTIONS[mask], data.shape)
    mask_matrix = numpy.logical_and(mask_matrix, dataarea_indicator, mask_matrix)   
    data = numpy.logical_xor(data, mask_matrix, data)
    
    if False:
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

def error_correction_and_reorder(values, version, format_info, version_info):
    CODEWORD_COUNT = [[19, 16, 13, 9], [34, 28, 22, 16], [55, 44, 34, 26], [80, 64, 48, 36], [108, 86, 62, 46], [136, 108, 76, 60], [156, 124, 88, 66], [194, 154, 110, 86], [232, 182, 132, 100], [274, 216, 154, 122], [324, 254, 180, 140], [370, 290, 206, 158], [428, 334, 244, 180], [461, 365, 261, 197], [523, 415, 295, 223], [589, 453, 325, 253], [647, 507, 367, 283], [721, 563, 397, 313], [795, 627, 445, 341], [861, 669, 485, 385], [932, 714, 512, 406], [1006, 782, 568, 442], [1094, 860, 614, 464], [1174, 914, 664, 514], [1276, 1000, 718, 538], [1370, 1062, 754, 596], [1468, 1128, 808, 628], [1531, 1193, 871, 661], [1631, 1267, 911, 701], [1735, 1373, 985, 745], [1843, 1455, 1033, 793], [1955, 1541, 1115, 845], [2071, 1631, 1171, 901], [2191, 1725, 1231, 961], [2306, 1812, 1286, 986], [2434, 1914, 1354, 1054], [2566, 1992, 1426, 1096], [2702, 2102, 1502, 1142], [2812, 2216, 1582, 1222], [2956, 2334, 1666, 1276]]
    BLOCK_COUNT = [[1 ,1 ,1 ,1] ,[1 ,1 ,1 ,1] ,[1 ,1 ,2 ,2] ,[1 ,2 ,2 ,4] ,[1 ,2 ,4 ,4] ,[2 ,4 ,4 ,4] ,[2 ,4 ,6 ,5] ,[2 ,4 ,6 ,6] ,[2 ,5 ,8 ,8] ,[4 ,5 ,8 ,8] ,[4 ,5 ,8 ,11] ,[4 ,8 ,10 ,11] ,[4 ,9 ,12 ,16] ,[4 ,9 ,16 ,16] ,[6 ,10 ,12 ,18] ,[6 ,10 ,17 ,16] ,[6 ,11 ,16 ,19] ,[6 ,13 ,18 ,21] ,[7 ,14 ,21 ,25] ,[8 ,16 ,20 ,25] ,[8 ,17 ,23 ,25] ,[9 ,17 ,23 ,34] ,[9 ,18 ,25 ,30] ,[10 ,20 ,27 ,32] ,[12 ,21 ,29 ,35] ,[12 ,23 ,34 ,37] ,[12 ,25 ,34 ,40] ,[13 ,26 ,35 ,42] ,[14 ,28 ,38 ,45] ,[15 ,29 ,40 ,48] ,[16 ,31 ,43 ,51] ,[17 ,33 ,45 ,54] ,[18 ,35 ,48 ,57] ,[19 ,37 ,51 ,60] ,[19 ,38 ,53 ,63] ,[20 ,40 ,56 ,66] ,[21 ,43 ,59 ,70] ,[22 ,45 ,62 ,74] ,[24 ,47 ,65 ,77] ,[25 ,49 ,68 ,81]]

    edc_level = [1 ,0 ,3 ,2][extract_int(format_info, 0, 2)]

    print edc_level, version, "edc level, version"

    block_count    = BLOCK_COUNT   [version - 1][edc_level]
    codeword_count = CODEWORD_COUNT[version - 1][edc_level]
    
    bytes_count = values.size / 8
        
    short_block_length = codeword_count / block_count
    long_block_count = codeword_count % block_count
    errorcorrection_block_length = (bytes_count - codeword_count) / block_count

    #print block_count, codeword_count, bytes_count, "block_count, codeword_count, bytes_count"
    #print short_block_length, long_block_count, errorcorrection_block_length, "short_block_length, long_block_count, errorcorrection_block_length"

    #if block_count > 1:
    short_block_indices = (range(block, short_block_length * block_count, block_count) for block in xrange(block_count - long_block_count))
    long_block_indices =  (range(block, short_block_length * block_count, block_count) + [shortblock_length * block_count + block] 
                            for block in xrange(block_count - long_block_count, long_block_count))
    datablock_indices =   (itertools.chain(short_block_indices,long_block_indices))
    #else:
    #    datablock_indices = [range(codeword_count)]
        
    correction_indices = (range(block + codeword_count, bytes_count, block_count) for block in xrange(block_count))

    #print datablock_indices
    #correction_indices = list(correction_indices)
    #print correction_indices

        

    packed = numpy.packbits(values) #reedsolo.rs_encode_msg
    #data = list((packed[block + correction_data], errorcorrection_block_length) for block, correction_data in itertools.izip(datablock_indices, correction_indices))

    corrected_bytes = list(reedsolo.rs_correct_msg(packed[block + correction_data], errorcorrection_block_length) for block, correction_data in itertools.izip(datablock_indices, correction_indices))
    #corrected_bytes = list(packed[block] for block in datablock_indices)

    #for corrected, extracted in itertools.izip(corrected_bytes, extracted_bytes) :
    #    print len(corrected), len(extracted)
    #    print corrected #numpy.array(corrected, numpy.int)
    #    print extracted


    #print corrected_bytes
    #print extracted_bytes   
        
    #extracted_byte_chain = numpy.fromiter(itertools.chain.from_iterable(extracted_bytes), dtype=numpy.uint8)
    corrected_byte_chain = numpy.fromiter(itertools.chain.from_iterable(corrected_bytes), dtype=numpy.uint8)    

    #print corrected_byte_chain.size
    #print extracted_byte_chain.size

    #for element in extractet_byte_chain :
    #    print element        

    #data = itertools.chain.from_iterable(apply_error_correction(packed[block],packed[correction_data]) for block, correction_data in itertools.izip(datablock_indices, correction_indices))
    
    corrected_bits = numpy.unpackbits(corrected_byte_chain)
    #print values.shape, values2.shape
    

    #print 
    #reordernum = 2   
    #if reordernum > 1:
    #    splitted = numpy.array_split(numpy.arange(0,values.size), numpy.arange(8,values.size, 8))
    #    reorderindices = list(itertools.chain.from_iterable(itertools.chain.from_iterable(splitted[i::reordernum] for i in xrange(reordernum))))
    #    print reorderindices
    #    values = values[reorderindices]

    return corrected_bits, version, format_info, version_info


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
    #print  ' '.join("{:02x}".format(value) for value in numpy.packbits(values)) , "pack"
    #print '""""""""""""""""""""""""""""""""""""""'
    
    #temp_int_list2 = numpy.append(temp_int_list[::2],temp_int_list[1::2])
    #print values[:40]
    #print ' '.join("{:02x}".format(value) for value in temp_int_list2)

    return int_list, mode_index
    #return temp_int_list2, mode_index


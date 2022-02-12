import ijson

fname = "/home/chris/Documents/UNI_neu/Masterarbeit/data_new/placetypes/en_debug_False/none_onlyorig_minwords50/filtered_dcm_tfidf_count.json"
CHUNKSIZE = 10

with open(fname, "r") as f:
    parser = ijson.parse(f)
    doit = False
    for prefix, event, value in parser:
        if (prefix, event) == ("used_config", "start_array"):
            doit = True
        elif (prefix, event) == ("used_config", "end_array"):
            doit = False
            break
        elif doit:
            print(value)

    #     if (prefix, event) == ('used_config', 'map_key'):
    #         stream.write('<%s>' % value)
    #         continent = value
    #     elif prefix.endswith('.name'):
    #         stream.write('<object name="%s"/>' % value)
    #     elif (prefix, event) == ('earth.%s' % continent, 'end_map'):
    #         stream.write('</%s>' % continent)
    # stream.write('</geo>')


# from bitstring import xrange
#
# def head(filename, lines=None, bytes=None):
#     if not lines and not bytes:
#         with open(filename, "r") as f:
#             return f.read()
#     elif lines and not bytes:
#         with open(filename, "r") as f:
#             lines = [f.readline() for line in xrange(1, lines+1)]
#             return filter(len, lines)
#     elif bytes and not lines:
#         string = ""
#         with open(filename, "r") as f:
#             while len(string) < bytes:
#                 string += f.read(CHUNKSIZE)
#         return string
# tmp = head(fname, bytes=2000)
# print()
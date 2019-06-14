import numpy

file = open('feat.txt', 'w')
file.write("// <feature name> <row> <col> <calculated temporal coherence value>\n")

i = 1
for row in range(204, 2448, 408):
    for col in range(204, 3264, 408):
        # if numpy.random.randint(10) >= 5: continue
        file.write('f{} {} {} 1\n'.format(i, row, col))
        i += 1

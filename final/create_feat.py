import numpy

file = open('feat.txt', 'w')
file.write("// <feature name> <row> <col> <calculated temporal coherence value>\n")

i = 1
for row in range(51, 2448, 102):
    for col in range(51, 3264, 102):
        if numpy.random.randint(10) >= 5: continue
        file.write('f{} {} {} 1\n'.format(i, row, col))
        i += 1

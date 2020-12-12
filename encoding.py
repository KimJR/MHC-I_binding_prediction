def A():
    return [-0.57, 0.39, -0.96, -0.61, -0.69]

def R():
    return [-0.40, -0.83, -0.61, 1.26, -0.28]

def N():
    return [-0.70, -0.63, -1.47, 1.02, 1.06]

def D():
    return [-1.62, -0.52, -0.67, 1.02, 1.47]

def C():
    return [0.07, 2.04, 0.65, -1.13, -0.39]

def Q():
    return [-0.05, -1.50, -0.67, 0.49, 0.21]

def E():
    return [-0.64, -1.59, -0.39, 0.69, 1.04]

def G():
    return [-0.90, 0.87, -0.36, 1.08, 1.95]

def H():
    return [0.73, -0.67, -0.42, 1.13, 0.99]

def I():
    return [0.59, 0.79, 1.44, -1.90, -0.93]

def L():
    return [0.65, 0.84, 1.25, -0.99, -1.90]

def K():
    return [-0.64, -1.19, -0.65, 0.68, -0.13]

def M():
    return [0.76, 0.05, 0.06, -0.62, -1.59]

def F():
    return [1.87, 1.04, 1.28, -0.61, -0.16]

def P():
    return [-1.82, -0.63, 0.32, 0.03, 0.68]

def S():
    return [-0.39, -0.27, -1.51, -0.25, 0.31]

def T():
    return [-0.04, -0.30, -0.82, -1.02, -0.04]

def W():
    return [1.38, 1.69, 1.91, 1.07, -0.05]

def Y():
    return [1.75, 0.11, 0.65, 0.21, -0.41]

def V():
    return [-0.02, 0.30, 0.97, -1.55, -1.16]

def BLOMAP(char):
    switcher = {
        'A': A(),
        'R': R(),
        'N': N(),
        'D': D(),
        'C': C(),
        'Q': Q(),
        'E': E(),
        'G': G(),
        'H': H(),
        'I': I(),
        'L': L(),
        'K': K(),
        'M': M(),
        'F': F(),
        'P': P(),
        'S': S(),
        'T': T(),
        'W': W(),
        'Y': Y(),
        'V': V(),
    }
    func = switcher.get(char, lambda: "Char is not a amino acid")
    return func

def encodeFile(file):
    for element in enumerate(file[0::]):
        for peptide in element[1][0:1:]:
            ascii = []
            for char in peptide:
                encoded = BLOMAP(char)
                ascii.append(encoded[0])
                ascii.append(encoded[1])
                ascii.append(encoded[2])
                ascii.append(encoded[3])
                ascii.append(encoded[4])
            file[element[0]][0] = ascii
    return file

def encodeInputFile(file):
    for x in range(len(file)):
        peptide = file[x][0]
        ascii = []
        for char in peptide:
            encoded = BLOMAP(char)
            ascii.append(encoded[0])
            ascii.append(encoded[1])
            ascii.append(encoded[2])
            ascii.append(encoded[3])
            ascii.append(encoded[4])
        file[x] = ascii
    return file

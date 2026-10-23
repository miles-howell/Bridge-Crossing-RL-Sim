NAME = "Beekeeper"

PALETTE = {
    "A": (64, 52, 44),     # outline
    "B": (186, 184, 172),  # suit shade
    "C": (240, 238, 228),  # suit
    "D": (112, 110, 102),  # veil mesh, dark
    "E": (172, 170, 160),  # veil mesh, light
    "H": (224, 168, 128),  # face behind the veil
    "J": (24, 20, 20),     # eyes, bee stripes
    "G": (232, 168, 40),   # gloves
    "I": (176, 112, 24),   # glove shade
    "Q": (248, 216, 48),   # bee
    "W": (40, 40, 48),     # leg and boot outline
    "X": (240, 238, 228),  # suit trousers
    "Y": (186, 184, 172),  # trouser seam
    "Z": (112, 116, 124),  # rubber boots
}

FRONT = [
    "......AAAA......",
    "....AACCCCAA....",
    "...ACCCCCCBBA...",
    "...ACCCCCCCBA...",
    "..AACCCCCCCBAA..",
    ".ACCCCCCCCCCCBA.",
    ".AAAAAAAAAAAAAA.",
    "..ADEDHHHHDEDA..",
    "..AEDHJHHJHDEA..",
    "..ADEHHHHHHEDA..",
    "..AEDEHHHHEDEA..",
    ".AADEDEDEDEDEAA.",
    ".ACCAAAAAAAACCA.",
    ".ACBACCCBCCABCA.",
    ".AGACCCCBCCCAGA.",
    ".AIACCCCBCCCAIA.",
    "..AACCCCBCCCAA..",
]
FRONT_WAVE = FRONT[:11] + [
    ".AADEDEDEDEDEAQJ",
    ".ACCAAAAAAAACAGA",
    ".ACBACCCBCCABAIA",
    ".AGACCCCBCCCAA..",
    ".AIACCCCBCCCA...",
    "..AACCCCBCCCAA..",
]
SIDE = [
    "......AAAA......",
    "....AACCCCA.....",
    "...ACCCCCCBA....",
    "...ACCCCCCBA....",
    "..AACCCCCCBAA...",
    ".ACCCCCCCCCCCA..",
    ".AAAAAAAAAAAAA..",
    "..ADEDEDHHHEA...",
    "..AEDEDHHJHHA...",
    "..ADEDEHHHHHA...",
    "..AEDEDEHHHEA...",
    "..ADEDEDEDEDA...",
    "...AACCCCCAA....",
    "....ACABCACA....",
    "....ACABCACA....",
    "....ACAGIACA....",
    "....ACCAACCA....",
]
BACK = FRONT[:7] + [
    "..ADEDEDEDEDEA..",
    "..AEDEDEDEDEDA..",
    "..ADEDEDEDEDEA..",
    "..AEDEDEDEDEDA..",
    ".AADEDEDEDEDEAA.",
    ".ACCAAAAAAAACCA.",
    ".ACBACCCCCCABCA.",
    ".AGACCCCCCCCAGA.",
    ".AIACCCBBCCCAIA.",
    "..AACCCBBCCCAA..",
]
BACK_WAVE = BACK[:11] + [
    "JQAADEDEDEDEDAA.",
    "AGACAAAAAAAACCA.",
    "AIABACCCCCCABCA.",
    "..AACCCCCCCCAGA.",
    "...ACCCBBCCCAIA.",
    "..AACCCBBCCCAA..",
]

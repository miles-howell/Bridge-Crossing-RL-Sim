NAME = "Sailor"

PALETTE = {
    "A": (24, 24, 48),     # outline
    "B": (36, 50, 104),    # navy shade
    "C": (56, 80, 152),    # navy
    "K": (240, 240, 232),  # white stripe
    "R": (216, 56, 48),    # neckerchief
    "S": (144, 32, 32),    # neckerchief knot
    "H": (248, 200, 160),  # skin
    "I": (216, 152, 120),  # skin shade
    "J": (24, 20, 20),     # eyes
    "O": (216, 112, 40),   # ginger hair
    "P": (160, 72, 24),    # hair shade
    "W": (20, 20, 36),     # leg and shoe outline
    "X": (56, 80, 152),    # navy trousers
    "Y": (36, 50, 104),    # trouser seam
    "Z": (44, 44, 52),     # black shoes
}

FRONT = [
    "................",
    "................",
    "..AAAAAAAAAAAA..",
    ".ACCCCCCCCCCCCA.",
    ".AACCCCKKCCCBAA.",
    "..AABBBBBBBBAA..",
    "..AAAAAAAAAAAA..",
    "..AOHHHHHHHHOA..",
    "..AOHJHHHHJHOA..",
    "..APHHHHHHHHPA..",
    "...AIHHHHHHIA...",
    "..AAARRSSRRAAA..",
    ".AKKAKRRRRKAKKA.",
    ".ACCACCRRCCACCA.",
    ".AHAKKKKKKKKAHA.",
    ".AIACCCCCCCCAIA.",
    "..AAKKKKKKKKAA..",
]
FRONT_WAVE = FRONT[:9] + [
    "..APHHHHHHHHPAA.",
    "...AIHHHHHHIAAHA",
    "..AAARRSSRRAAKIA",
    ".AKKAKRRRRKAKKA.",
    ".ACCACCRRCCAAA..",
    ".AHAKKKKKKKKA...",
    ".AIACCCCCCCCA...",
    "..AAKKKKKKKKAA..",
]
SIDE = [
    "................",
    "................",
    "...AAAAAAAAA....",
    "..ACCCCCCCCCA...",
    "..AABBBBBBBBA...",
    "...AAAAAAAAAAAA.",
    "...AOOHHHHHHA...",
    "...AOIHHHHJHA...",
    "...AOOHHHHHHHA..",
    "....APHHHHHHA...",
    ".....AIHHHHA....",
    "....AARRRSA.....",
    "...AAKKKKKRA....",
    "....ACACCARA....",
    "....AKAKKAKA....",
    "....ACAHIACA....",
    "....AKKAAKKA....",
]
BACK = [
    "................",
    "................",
    "..AAAAAAAAAAAA..",
    ".ACCCCCCCCCCCCA.",
    ".AACCCCCCCCCBAA.",
    "..AABBBBBBBBAA..",
    "..AOOOOOOOOOOA..",
    "..AOOOOOOOOOOA..",
    "..APOOOOOOOOPA..",
    "..APPOOOOOOPPA..",
    "...AIHHHHHHIA...",
    "..AAAKKKKKKAAA..",
    ".AKKAKCCCCKAKKA.",
    ".ACCAKKKKKKACCA.",
    ".AHAKKKKKKKKAHA.",
    ".AIACCCCCCCCAIA.",
    "..AAKKKKKKKKAA..",
]
BACK_WAVE = BACK[:9] + [
    ".AAPPOOOOOOPPA..",
    "AHAAIHHHHHHIA...",
    "AIKAAKKKKKKAAA..",
    ".AKKAKCCCCKAKKA.",
    "..AAAKKKKKKACCA.",
    "...AKKKKKKKKAHA.",
    "...ACCCCCCCCAIA.",
    "..AAKKKKKKKKAA..",
]

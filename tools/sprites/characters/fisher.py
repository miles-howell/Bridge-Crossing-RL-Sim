NAME = "Fisher"

PALETTE = {
    "A": (44, 38, 24),     # outline
    "K": (204, 180, 124),  # bucket hat
    "L": (150, 126, 80),   # hat shade and band
    "H": (212, 150, 100),  # skin
    "I": (168, 108, 68),   # skin shade
    "J": (24, 20, 20),     # eyes
    "S": (132, 112, 92),   # stubble and hair
    "C": (238, 228, 196),  # shirt
    "B": (196, 182, 148),  # shirt shade
    "O": (112, 128, 52),   # waders
    "P": (68, 80, 30),     # wader straps and shade
    "R": (124, 82, 40),    # fishing rod
    "W": (36, 38, 20),     # leg and boot outline
    "X": (112, 128, 52),   # wader legs
    "Y": (68, 80, 30),     # wader seam
    "Z": (72, 60, 44),     # wader boots
}

HAT = [
    ".....AAAAAA.....",
    "....AKKKKKKA....",
    "....AKKKKKLA....",
    "....ALLLLLLA....",
    "...AKKKKKKKKA...",
    "..AKKKKKKKKKLA..",
    ".AKLAAAAAAAALKA.",
]

FRONT = HAT + [
    "..AAAHHHHHHAAA..",
    "...AIHJHHJHIA...",
    "...AIHHHHHHIA...",
    "....ASHHHHSA....",
    ".....ASSSSA.....",
    "..ACCCAIIACCCA..",
    ".ACBPCOOOOCPBCA.",
    ".AHAPOOOOOOPAHA.",
    ".AIAOOOOOOOOAIA.",
    "..AAOOOOOOOOAA..",
]
FRONT_WAVE = FRONT[:11] + [
    ".....ASSSSA..AHA",
    "..ACCCAIIACCCACA",
    ".ACBPCOOOOCPBACA",
    ".AHAPOOOOOOPAA..",
    ".AIAOOOOOOOOA...",
    "..AAOOOOOOOOAA..",
]
SIDE = HAT[:6] + [
    ".AKLAAAAAAAAKKA.",
    "..AAAIHHHHHHAA..",
    "....AIHHHJHHA...",
    "....AIHHHHHHHA..",
    "....AISHHSSA....",
    ".....ASSSSA.....",
    "....ACCAIIAA....",
    "....AOACBAPA....",
    "....AOACBAOA....",
    "....AOAHIAOA....",
    "....AOOAAOOA....",
]

# The rod rests on the right shoulder and rises past the hat brim.
ROD = [(3, 15), (4, 15), (5, 15), (6, 15), (7, 14), (8, 14), (9, 13), (10, 13), (11, 12), (12, 12)]


def with_rod(lines):
    rows = [list(r) for r in lines]
    for y, x in ROD:
        rows[y][x] = "R"
    return ["".join(r) for r in rows]


BACK = with_rod(HAT + [
    "..AAASSSSSSAAA..",
    "...ASSSSSSSSA...",
    "...AIHHHHHHIA...",
    "....AIHHHHIA....",
    ".....AIIIIA.....",
    "..ACCCAAAACCCA..",
    ".ACBPCCCCCCPBCA.",
    ".AHAPCCCCCCPAHA.",
    ".AIAOOOOOOOOAIA.",
    "..AAOOOOOOOOAA..",
])
BACK_WAVE = with_rod(BACK[:11] + [
    "AHA..AIIIIA.....",
    "ACACCCAAAACCCA..",
    "ACABPCCCCCCPBCA.",
    "..AAPCCCCCCPAHA.",
    "...AOOOOOOOOAIA.",
    "..AAOOOOOOOOAA..",
])

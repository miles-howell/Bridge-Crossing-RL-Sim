NAME = "Aviator"

PALETTE = {
    "A": (40, 24, 36),     # outline
    "B": (92, 44, 92),     # jacket shade
    "C": (136, 68, 128),   # plum leather jacket
    "K": (196, 140, 184),  # collar
    "L": (128, 80, 44),    # leather cap
    "M": (88, 52, 30),     # cap shade, ear flaps
    "N": (216, 168, 64),   # goggle rims
    "O": (164, 212, 236),  # goggle lenses
    "S": (242, 230, 200),  # scarf
    "T": (200, 182, 146),  # scarf shade
    "H": (204, 150, 104),  # skin
    "I": (164, 112, 76),   # skin shade
    "F": (104, 62, 44),    # skin outline
    "J": (24, 20, 20),     # eyes
    "G": (128, 80, 44),    # gloves
    "W": (36, 24, 20),     # leg and boot outline
    "X": (128, 92, 60),    # trousers
    "Y": (92, 62, 40),     # trouser seam
    "Z": (72, 44, 28),     # boots
}

FRONT = [
    "................",
    "................",
    ".....AAAAAA.....",
    "...AALLLLLLAA...",
    "..ALLLLLLLLLMA..",
    "..ALNNNLLNNNMA..",
    "..ANOONNNNOONA..",
    "..ALNNNLLNNNMA..",
    "..ALLLLLLLLLMA..",
    "..ALHJHHHHJHMA..",
    "..AMHHHIIHHHMA..",
    "..AMAHHHHHHAMA..",
    ".ACKKSSSSSSKKCA.",
    ".ACBAKSSSTKABCA.",
    ".AGACCSTTCCCAGA.",
    ".AMACCSTBCCCAMA.",
    "..AACCCTBCCCAA..",
]
FRONT_WAVE = FRONT[:9] + [
    "..ALHJHHHHJHMAGA",
    "..AMHHHIIHHHMAMA",
    "..AMAHHHHHHAMACA",
    ".ACKKSSSSSSKKCCA",
    ".ACBAKSSSTKABAA.",
    ".AGACCSTTCCCAA..",
    ".AMACCSTBCCCA...",
    "..AACCCTBCCCAA..",
]
SIDE = [
    "................",
    "................",
    ".....AAAAA......",
    "....ALLLLLAA....",
    "...ALLLLLNNNA...",
    "...ALLLLNOONA...",
    "...ALLLLLNNNA...",
    "...ALLLLLLLLA...",
    "...ALMLHHHHHA...",
    "...ALMLHHHJHHA..",
    "...AMMLHHHHHA...",
    "....AMAHHHHIA...",
    "..ASSKKSSSKA....",
    ".ASTACABCSCA....",
    "..ATACABCACA....",
    "....ACAGMACA....",
    "....ACCAACCA....",
]
BACK = [
    "................",
    "................",
    ".....AAAAAA.....",
    "...AALLLLLLAA...",
    "..ALLLLLLLLLMA..",
    "..AMMMMNNMMMMA..",
    "..ALLLLLLLLLMA..",
    "..ALLLLLLLLLMA..",
    "..ALLLLLLLLLMA..",
    "..ALLLLLLLLLMA..",
    "..AMLLLLLLLLMA..",
    "..AMAFHHHHFAMA..",
    ".ACKKSSSSSSKKCA.",
    ".ACBAKSSSSKABCA.",
    ".AGACCCSTCCCAGA.",
    ".AMACCCSTCCCAMA.",
    "..AACCCTTCCCAA..",
]
BACK_WAVE = BACK[:9] + [
    "AGALLLLLLLLLMA..",
    "AMAMLLLLLLLLMA..",
    "ACAMAFHHHHFAMA..",
    "ACCKKSSSSSSKKCA.",
    ".AABAKSSSSKABCA.",
    "..AACCCSTCCCAGA.",
    "...ACCCSTCCCAMA.",
    "..AACCCTTCCCAA..",
]

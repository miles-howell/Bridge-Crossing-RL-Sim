NAME = "Engineer"

PALETTE = {
    "A": (48, 32, 28),     # outline
    "K": (44, 32, 30),     # hair
    "L": (76, 56, 50),     # hair highlight
    "H": (176, 116, 76),   # skin
    "I": (136, 84, 56),    # skin shade
    "J": (32, 28, 36),     # glasses rims
    "O": (216, 236, 248),  # lens glint
    "M": (64, 52, 60),     # eyes behind the lenses
    "Q": (240, 200, 56),   # pencil, overall buttons
    "S": (244, 228, 148),  # shirt
    "T": (204, 184, 104),  # shirt shade
    "U": (76, 116, 180),   # denim
    "V": (48, 76, 132),    # denim shade
    "P": (156, 204, 240),  # blueprint roll
    "W": (36, 28, 28),     # leg and boot outline
    "X": (76, 116, 180),   # denim trousers
    "Y": (48, 76, 132),    # trouser seam
    "Z": (112, 70, 44),    # boots
}

FRONT = [
    "................",
    "....AA.AA.AA....",
    "...AKKAKKAKKA...",
    "..AKLKKLKKLKKA..",
    "..AKKKLKKKLKKA..",
    ".AKLKKKKLKKKLKA.",
    "QAKKHHHHHHHHKKA.",
    "AQIHJJHHHHJJHIA.",
    ".AHJOMJHHJMOJHA.",
    ".AIHJJHHHHJJHIA.",
    "..AIHHHIIHHHIA..",
    "...AIIHHHHIIA...",
    ".ASSAUAAAAUASSA.",
    ".ASTAUQUUQUATSA.",
    ".AHAUUUVVUUUAHA.",
    ".AIAUUUVVUUUAIA.",
    "..AAUUUVVUUUAA..",
]
FRONT_WAVE = FRONT[:8] + [
    ".AHJOMJHHJMOJHAA",
    ".AIHJJHHHHJJHIAP",
    "..AIHHHIIHHHIAAP",
    "...AIIHHHHIIA.AP",
    ".ASSAUAAAAUASAHP",
    ".ASTAUQUUQUATAIA",
    ".AHAUUUVVUUUAA..",
    ".AIAUUUVVUUUA...",
    "..AAUUUVVUUUAA..",
]
SIDE = [
    "................",
    ".....AA.AA.A....",
    "....AKKAKKAKA...",
    "...AKLKKLKKLKA..",
    "..AKKKLKKKLKKKA.",
    "..AKLKKKLKKKKHA.",
    "..AKKKKQKHHHHHA.",
    "..AKKKQIHHJJHHA.",
    "..AKKKIJJJOMJHA.",
    "..AKKKKIHHJJHHA.",
    "...AKKHHHHHHHA..",
    "....AIIHHHHHA...",
    "....ASSUSSSA....",
    "....AUASTAUA....",
    "....AUASTAQA....",
    "....AUAHIAUA....",
    "....AUUAAUUA....",
]
BACK = [
    "................",
    "....AA.AA.AA....",
    "...AKKAKKAKKA...",
    "..AKLKKLKKLKKA..",
    "..AKKKLKKKLKKA..",
    ".AKLKKKKLKKKLKA.",
    ".AKKKLKKKKLKKKAQ",
    ".AKLKKKLKKKLKQA.",
    ".AIKKLKKKLKKKIA.",
    ".AIKKKKLKKKKKIA.",
    "..AKKKKKKKKKKA..",
    "...AAIHHHHIAA...",
    ".ASSAUAAAAUASSA.",
    ".ASTAVUUUUVATSA.",
    ".AHAUVUUUUVUAHA.",
    ".AIAUUUUUUUUAIA.",
    "..AAUUUVVUUUAA..",
]
BACK_WAVE = BACK[:8] + [
    "AAIKKLKKKLKKKIA.",
    "PAIKKKKLKKKKKIA.",
    "PAAKKKKKKKKKKA..",
    "PA.AAIHHHHIAA...",
    "PHASAUAAAAUASSA.",
    "AIATAVUUUUVATSA.",
    "..AAUVUUUUVUAHA.",
    "...AUUUUUUUUAIA.",
    "..AAUUUVVUUUAA..",
]

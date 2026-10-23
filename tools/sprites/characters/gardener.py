NAME = "Gardener"

PALETTE = {
    "A": (60, 36, 28),     # outline
    "S": (236, 204, 116),  # straw
    "T": (188, 148, 68),   # straw shade, hat band
    "R": (216, 48, 48),    # flower
    "K": (36, 28, 28),     # hair
    "H": (140, 86, 56),    # skin
    "I": (104, 60, 40),    # skin shade
    "J": (20, 16, 16),     # eyes
    "C": (240, 228, 200),  # shirt
    "B": (200, 186, 156),  # shirt shade
    "P": (232, 120, 160),  # apron
    "Q": (184, 80, 120),   # apron shade, pocket
    "G": (164, 168, 176),  # trowel blade
    "N": (120, 72, 40),    # trowel handle
    "W": (40, 30, 24),     # leg and boot outline
    "X": (128, 84, 52),    # trousers
    "Y": (92, 58, 36),     # trouser seam
    "Z": (52, 100, 60),    # rubber boots
}

FRONT = [
    "................",
    "................",
    ".....AAAAAA.....",
    "....ASSSSSTA....",
    "..AAATTTTTTAAA..",
    ".ASSSSSSSSSSRSA.",
    "ASSSSSSSSSSSSSTA",
    ".AAAAAAAAAAAAAA.",
    "...AKIIIIIIKA...",
    "...AKHJHHJHKA...",
    "...AIHHHHHHIA...",
    "...AAIHHHHIAA...",
    ".ACCACCPPCCACCA.",
    ".ACBACPPPPCABCA.",
    ".AHACPPPPPPCAHA.",
    ".AIACPPQQPPCAIA.",
    "..AAPPPPPPPPAA..",
]
FRONT_WAVE = FRONT[:8] + [
    "...AKIIIIIIKA.A.",
    "...AKHJHHJHKAAGA",
    "...AIHHHHHHIAAGA",
    "...AAIHHHHIAAANA",
    ".ACCACCPPCCACAHA",
    ".ACBACPPPPCABAIA",
    ".AHACPPPPPPCAA..",
    ".AIACPPQQPPCA...",
    "..AAPPPPPPPPAA..",
]
SIDE = [
    "................",
    "................",
    ".....AAAAA......",
    "....ASSSSTA.....",
    "..AAATTTTTAAA...",
    ".ASSSSSSSRSSSA..",
    "ASSSSSSSSSSSSSTA",
    ".AAAAAAAAAAAAAA.",
    "...AKKIIIIIIA...",
    "...AKKIHHHJHHA..",
    ".AKKAKIHHHHHA...",
    "..AA.AAIHHIA....",
    "...AACCCCPPA....",
    "....ACBBCCPA....",
    "....ACBBCCPA....",
    "....ACHICCPA....",
    "....ACCAAPPA....",
]
BACK = [
    "................",
    "................",
    ".....AAAAAA.....",
    "....ATSSSSSA....",
    "..AAATTTTTTAAA..",
    ".ASRSSSSSSSSSA..",
    "ATSSSSSSSSSSSSSA",
    ".AAAAAAAAAAAAAA.",
    "...AKKKKKKKKA...",
    "...AKKKKKKKKA...",
    "...AIKKKKKKIA...",
    "....AIAKKAIA....",
    ".ACCACCAACCACCA.",
    ".ACBACPCCPCABCA.",
    ".AHACCCPPCCCAHA.",
    ".AIAPPQPPQPPAIA.",
    "..AACCCCCCCCAA..",
]
BACK_WAVE = BACK[:8] + [
    ".A.AKKKKKKKKA...",
    "AGAAKKKKKKKKA...",
    "AGAAIKKKKKKIA...",
    "ANA.AIAKKAIA....",
    "AHACACCAACCACCA.",
    "AIABACPCCPCABCA.",
    "..AACCCPPCCCAHA.",
    "...APPQPPQPPAIA.",
    "..AACCCCCCCCAA..",
]

NAME = "Builder"

PALETTE = {
    "A": (40, 28, 24),     # outline
    "B": (192, 192, 196),  # hard hat shade
    "C": (244, 244, 240),  # hard hat
    "F": (64, 36, 24),     # skin outline
    "H": (124, 76, 48),    # skin
    "I": (92, 54, 34),     # skin shade
    "J": (16, 12, 12),     # eyes
    "K": (28, 24, 24),     # hair
    "M": (150, 150, 160),  # wrench
    "O": (240, 120, 32),   # hi-vis vest
    "P": (184, 76, 20),    # vest shade
    "R": (232, 236, 200),  # reflective stripe
    "S": (84, 84, 92),     # work shirt
    "W": (28, 28, 32),     # leg and boot outline
    "X": (72, 72, 80),     # trousers
    "Y": (48, 48, 56),     # trouser seam
    "Z": (168, 112, 56),   # work boots
}

FRONT = [
    ".....AAAAAA.....",
    "....ACCCBCCA....",
    "...ACCCCBCCBA...",
    "...ACCCCBCCBA...",
    "..AACCCCBCCBAA..",
    "..ACCCCCCCCCBA..",
    "..AAAAAAAAAAAA..",
    "...AKHHHHHHKA...",
    "...AHHJHHJHHA...",
    "...AIHHHHHHIA...",
    "....AIHIIHIA....",
    ".ASSAAAIIAAASSA.",
    ".ASAOOOSSOOOASA.",
    ".ASARRRRRRRRASA.",
    ".AHAOOOSSOOOAHA.",
    ".AIAPOOSSOOPAIA.",
    "..AAPPPSSPPPAA..",
]
FRONT_WAVE = FRONT[:9] + [
    "...AIHHHHHHIA.M.",
    "....AIHIIHIA.AMA",
    ".ASSAAAIIAAASAHA",
    ".ASAOOOSSOOOASSA",
    ".ASARRRRRRRRASA.",
    ".AHAOOOSSOOOAA..",
    ".AIAPOOSSOOPA...",
    "..AAPPPSSPPPAA..",
]
SIDE = [
    "......AAAA......",
    "....AACCCBA.....",
    "...ACCCCCCBA....",
    "...ACCCCCCBA....",
    "...ACCCCCCCBA...",
    "..ABBCCCCCCCCCA.",
    "...AAAAAAAAAAA..",
    "...AKKHHHHHHA...",
    "...AKKHHHHJHHA..",
    "...AKHHHHHHHA...",
    "....AIHHHIIA....",
    ".....AAIIAA.....",
    "...AAOOOOOSAA...",
    "....AOASSAOA....",
    "....ARASSARA....",
    "....AOAHIAOA....",
    "....APPAAPPA....",
]
BACK = FRONT[:7] + [
    "...AKKKKKKKKA...",
    "...AKKKKKKKKA...",
    "...AHIHHHHIHA...",
    "....AIHHHHIA....",
    ".ASSAAIIIIAASSA.",
    ".ASAOOOOOOOOASA.",
    ".ASARRRRRRRRASA.",
    ".AHAOOOOOOOOAHA.",
    ".AIAPOOOOOOPAIA.",
    "..AAPPPPPPPPAA..",
]
BACK_WAVE = BACK[:9] + [
    ".M.AHIHHHHIHA...",
    "AMA.AIHHHHIA....",
    "AHASAAIIIIAASSA.",
    "ASSAOOOOOOOOASA.",
    ".ASARRRRRRRRASA.",
    "..AAOOOOOOOOAHA.",
    "...APOOOOOOPAIA.",
    "..AAPPPPPPPPAA..",
]

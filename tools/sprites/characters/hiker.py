NAME = "Hiker"

PALETTE = {
    "A": (60, 36, 30),     # outline
    "R": (170, 76, 42),    # hair
    "S": (118, 48, 28),    # hair shade
    "T": (214, 120, 70),   # hair highlight
    "H": (236, 186, 142),  # skin
    "I": (198, 140, 104),  # skin shade
    "J": (24, 20, 20),     # eyes
    "C": (62, 158, 152),   # jacket
    "B": (34, 104, 108),   # jacket shade
    "K": (132, 206, 192),  # zip, jacket highlight
    "P": (178, 90, 42),    # backpack and straps
    "Q": (122, 58, 30),    # backpack shade
    "N": (98, 120, 160),   # sleeping mat
    "O": (64, 80, 116),    # sleeping mat shade
    "W": (48, 34, 28),     # leg and boot outline
    "X": (196, 172, 116),  # khaki trousers
    "Y": (148, 124, 80),   # trouser seam
    "Z": (116, 66, 36),    # hiking boots
}

FRONT = [
    "................",
    ".....AAAAAA.....",
    "...AARRRTTRAA...",
    "..ARRRRTTRRRRA..",
    "..ARRRRRRRRRSA..",
    ".ARSRRRRRRRRSRA.",
    ".ARSHHRRRRHHSRA.",
    ".ASHHHHHHHHHHSA.",
    ".AIHHJHHHHJHHIA.",
    "..AIHHHIIHHHIA..",
    "..AIHHHHHHHHIA..",
    "...AAIHHHHIAA...",
    "..ABPCCCCCCPBA..",
    ".ABCPCCKBCCPCBA.",
    ".AHAPCCKBCCPAHA.",
    ".AIAPBCKBCBPAIA.",
    "..AACCCKBCCCAA..",
]
FRONT_WAVE = FRONT[:11] + [
    "...AAIHHHHIAAAHA",
    "..ABPCCCCCCPBAIA",
    ".ABCPCCKBCCPCBCA",
    ".AHAPCCKBCCPCAA.",
    ".AIAPBCKBCBPA...",
    "..AACCCKBCCCAA..",
]
SIDE = [
    "................",
    ".....AAAAAA.....",
    "...AARRRRTTAA...",
    "..ARRRRRRRRTRA..",
    "..ARRRRRRRRRRA..",
    "..ARSRRRRRRRRRA.",
    ".AARSRRRRRHHHHA.",
    "ARARSRRRIHHHJHA.",
    "ASAARSRRIHHHHHHA",
    ".A..ASRIIHHHHHA.",
    "AAAAAASIIHHHHA..",
    "ANNNOAAAIHHHAA..",
    "AOOOOAPCCCCCA...",
    "APPPQAPABCCKA...",
    "APQPQAPABCCCA...",
    "APPPQAPAHBCCA...",
    ".AAAAACCAACCA...",
]
BACK = [
    "................",
    ".....AAAAAA.....",
    "...AARRRRTTAA...",
    "..ARRRRRRRRTRA..",
    "..ARRRRRRRRRRA..",
    ".ARRRRRRRRRRSRA.",
    ".ARSRRRRRRRRSRA.",
    ".ARSRRRRRRRRSRA.",
    ".AIRSRRRRRRSRIA.",
    "..AIRSRSSRSRIA..",
    "...AIIARRAIIA...",
    ".AANNNARSANNNAA.",
    ".ACOOOOAAOOOOCA.",
    ".ACAPPPPPPPPACA.",
    ".AHAPPQQQQPPAHA.",
    ".AIAPPQPPQPPAIA.",
    "..AAPPQQQQPPAA..",
]
BACK_WAVE = BACK[:11] + [
    "AHAANNARSANNNAA.",
    "AIAOOOOAAOOOOCA.",
    "ACBAPPPPPPPPACA.",
    ".AAAPPQQQQPPAHA.",
    "...APPQPPQPPAIA.",
    "..AAPPQQQQPPAA..",
]

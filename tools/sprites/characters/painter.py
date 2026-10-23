NAME = "Painter"

PALETTE = {
    "A": (56, 32, 24),     # outline
    "B": (184, 136, 32),   # smock shade
    "C": (228, 184, 60),   # smock
    "R": (136, 36, 52),    # beret
    "S": (92, 22, 36),     # beret shade
    "K": (180, 64, 80),    # beret highlight
    "H": (248, 212, 180),  # skin
    "I": (224, 168, 136),  # skin shade
    "N": (80, 46, 28),     # hair
    "J": (24, 20, 20),     # eyes
    "P": (208, 48, 48),    # red paint
    "U": (56, 100, 204),   # blue paint
    "T": (150, 96, 48),    # brush handle
    "W": (32, 24, 24),     # leg and shoe outline
    "X": (100, 66, 42),    # trousers
    "Y": (66, 42, 26),     # trouser seam
    "Z": (56, 52, 52),     # shoes
}

FRONT = [
    ".........AA.....",
    ".....AAAARRA....",
    "...AARRKKRRRA...",
    "..ARRRKRRRRRRA..",
    ".ARRRRRRRRRRRRA.",
    "ARRRSSSSSSSSSSA.",
    "ASSAANNNNNNNNA..",
    ".AAANHHHHHHNNA..",
    "..ANHHJHHJHHNA..",
    "..AIHHHHHHHHIA..",
    "...AIHHIIHHIA...",
    "...AAIHHHHIAA...",
    ".ACCCAAHHAACCCA.",
    ".ACBACCCCCCABCA.",
    ".AHACPCCCCUCAHA.",
    ".AIACCCCCCCCAIA.",
    "..AACCUCCPCCAA..",
]
FRONT_WAVE = FRONT[:8] + [
    "..ANHHJHHJHHNAP.",
    "..AIHHHHHHHHIAP.",
    "...AIHHIIHHIA.T.",
    "...AAIHHHHIAA.T.",
    ".ACCCAAHHAACCAHA",
    ".ACBACCCCCCABAIA",
    ".AHACPCCCCUCAA..",
    ".AIACCCCCCCCA...",
    "..AACCUCCPCCAA..",
]
SIDE = [
    "......AA........",
    "...AAAARRA......",
    "..ARRRKKRRA.....",
    ".ARRRRRKRRRA....",
    "ARRRRRRRRRRRA...",
    "ARSSSSSSSSSA....",
    "ASAANNNNNHHA....",
    ".AAANNNHHHHHA...",
    "..ANNNHHHHJHA...",
    "..ANNNIHHHHHHA..",
    "...ANNHHHHHIA...",
    "....AAIHHIAA....",
    "....AACCCCAA....",
    "....ACABCAPA....",
    "....ACABCACA....",
    "....ACAHIACA....",
    "....ACCAAUCA....",
]
BACK = [
    ".....AA.........",
    "....ARRAAAA.....",
    "...ARRRKKRRAA...",
    "..ARRRRRRKRRRA..",
    ".ARRRRRRRRRRRRA.",
    ".ASSSSSSSSSSRRRA",
    "..ANNNNNNNNAASSA",
    "..ANNNNNNNNNNAA.",
    "..ANNNNNNNNNNA..",
    "..ANNNNNNNNNNA..",
    "...ANNNNNNNNA...",
    "...AAIHHHHIAA...",
    ".ACCCAAAAAACCCA.",
    ".ACBACCCCCCABCA.",
    ".AHACCCCCCCCAHA.",
    ".AIACCCPCCCCAIA.",
    "..AACCCCCUCCAA..",
]
BACK_WAVE = BACK[:8] + [
    ".PANNNNNNNNNNA..",
    ".PANNNNNNNNNNA..",
    ".T.ANNNNNNNNA...",
    ".T.AAIHHHHIAA...",
    "AHACCAAAAAACCCA.",
    "AIABACCCCCCABCA.",
    "..AACCCCCCCCAHA.",
    "...ACCCPCCCCAIA.",
    "..AACCCCCUCCAA..",
]

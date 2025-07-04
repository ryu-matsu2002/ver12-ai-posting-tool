from enum import Enum

class BlogType(str, Enum):
    NOTE = "note"
    AMEBA = "ameblo"
    LIVEDOOR = "livedoor"
    HATENA = "hatena"
    SEESAA = "seesaa"
    EXCITE = "excite"

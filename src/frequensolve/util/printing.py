"""Terminal color helpers for FrequenSolve status messages."""

__all__ = ["ANSIColorCodes"]


def print_note(msg: str):
    """Print a styled note message.

    Args:
        msg: Message text.
    """

    print(f"{ANSIColorCodes.note}Note: {msg}{ANSIColorCodes.none}")


def print_warn(msg: str):
    """Print a styled warning message.

    Args:
        msg: Message text.
    """

    print(f"{ANSIColorCodes.warn}Warning: {msg}{ANSIColorCodes.none}")


class ANSIColorCodes:
    """ANSI escape-code constants used by simple terminal output helpers."""

    # Simple colors
    fg_Black = "\033[30m"
    fg_Red = "\033[31m"
    fg_Green = "\033[32m"
    fg_Yellow = "\033[33m"
    fg_Blue = "\033[34m"
    fg_Magenta = "\033[35m"
    fg_Cyan = "\033[36m"
    fg_White = "\033[37m"
    fg_br_Black = "\033[90m"
    fg_br_Red = "\033[91m"
    fg_br_Green = "\033[92m"
    fg_br_Yellow = "\033[93m"
    fg_br_Blue = "\033[94m"
    fg_br_Magenta = "\033[95m"
    fg_br_Cyan = "\033[96m"
    fg_br_White = "\033[97m"
    bg_Black = "\033[40m"
    bg_Red = "\033[41m"
    bg_Green = "\033[42m"
    bg_Yellow = "\033[43m"
    bg_Blue = "\033[44m"
    bg_Magenta = "\033[45m"
    bg_Cyan = "\033[46m"
    bg_White = "\033[47m"
    bg_br_Black = "\033[100m"
    bg_br_Red = "\033[101m"
    bg_br_Green = "\033[102m"
    bg_br_Yellow = "\033[103m"
    bg_br_Blue = "\033[104m"
    bg_br_Magenta = "\033[105m"
    bg_br_Cyan = "\033[106m"
    bg_br_White = "\033[107m"

    # Used colors
    none = "\033[0m"
    warn = fg_br_Yellow
    note = "\033[38;5;139m"
    faint = "\033[38;5;244m"
    error = fg_br_Red

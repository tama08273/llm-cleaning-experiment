from .basic_cleaner import basic_cleaner
from .advanced_cleaner import advanced_cleaner

def get_cleaner(name):
    if name == "basic":
        return basic_cleaner()
    elif name == "advanced":
        return advanced_cleaner()
    else:
        return None
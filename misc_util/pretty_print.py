from IPython.display import Markdown, display
from IPython import get_ipython
#TODO if importerror just dont try ipython stuff

class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


TRANSLATOR = {
    "**": color.BOLD,
    "__": color.UNDERLINE,
    "*p*": color.PURPLE,
    "*c*": color.CYAN,
    "*d*": color.DARKCYAN,
    "*b*": color.BLUE,
    "*g*": color.GREEN,
    "*y*": color.YELLOW,
    "*r*": color.RED,
    "*end*": color.END,
}

JUPYTER_TRANSLATOR = {
    "**": '<span style="font-weight:bold">',
    "__": '<span style="text-decoration:underline">',
    "*p*": '<span style="color: #ff00ff">',
    "*c*": '<span style="color: #00ffff">',
    "*d*": '<span style="color: #009999">',
    "*b*": '<span style="color: #0000ff">',
    "*g*": '<span style="color: #00ff00">',
    "*y*": '<span style="color: #ffff00">',
    "*r*": '<span style="color: #ff0000">',
    "*end*": '</span>',
}


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def pretty_print(*args, fontsize=10, **kwargs):
    """https://stackoverflow.com/questions/8924173/how-do-i-print-bold-text-in-python/8930747"""
    if isnotebook():
        display(Markdown(f'<span style="font-size:{fontsize}pt">'+fmt(*args, isnotebook=True)+'</span>'))
    else:
        # TODO: check if os is linux else just remove ;)
        print(fmt(*args), **kwargs)

def fmt(*args, isnotebook=False):
    translator = JUPYTER_TRANSLATOR if isnotebook else TRANSLATOR
    to_print = " ".join(str(i) for i in args)
    for orig, rep in translator.items():
        to_print = to_print.split(orig)
        to_print[1::2] = [rep + i + translator["*end*"] for i in to_print[1::2]]
        to_print = "".join(to_print)
    return to_print


def print_multicol(lst, line_len=220):
    max_len = (max(len(i) for i in lst) + 1)
    n_cols = max(line_len//max_len, 1)
    divisions = list(zip(*[lst[i::n_cols] for i in range(n_cols)]))
    for elems in zip(divisions):
        print("".join([i.ljust(max_len) for i in elems[0]]))


if __name__ == "__main__":
    pretty_print(
        "Hello this is **bold** and **this is too**, __and this underlined__, __**and this both**__ and *r*this is red*r*"
    )

from pylab import *

def gen_figure(pytex, scale = 0.8, fig_num = None):
    current_pytex = pytex
    fig_info = {
        "dir": "./figures/generated/",
        "basename": '_'.join([pytex.session, pytex.family, pytex.restart]),
        "num": 0
    }
    pytex.fig_info = fig_info

    nums = []
    if fig_num != None:
        nums.append(fig_num)
    else:
        nums = get_fignums()
    fig_info = pytex.fig_info
    results = ""
    for num in nums:
        name = fig_info["basename"] + "_fig" + str(fig_info["num"]) + ".pdf"
        path = fig_info["dir"] + name
        fig_info["num"] = fig_info["num"] + 1
        figure(num)
        savefig(path)
        close(num)
        pytex.add_created(path)
        results = results + "\\includegraphics[scale={}]".format(scale) + "{" + path + "}\n\n"
    return results
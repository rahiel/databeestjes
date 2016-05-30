import seaborn as sns

rgb2hex = lambda x: '#' + ''.join([hex(int(c * 255))[2:] for c in x])


def colorize_html_text(text, palette="hls"):
    out = []
    colors = [rgb2hex(c) for c in sns.color_palette(palette, n_colors=len(text))]
    for color, char in zip(colors, text):
        out.append('<span style="color: %s";>%s</span>' % (color, char))

    return "".join(out)


print(colorize_html_text("Databeestjes - 91", "hls"))

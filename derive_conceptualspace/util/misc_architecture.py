import sys
from datetime import datetime

from misc_util.logutils import CustomIO


def merge_streams(s1, s2, for_):
    format = sys.stdout.date_format if isinstance(sys.stdout, CustomIO) else CustomIO.DEFAULT_DATE_FORMAT
    if not s1 and not s2:
        return ""

    def make_list(val):
        res = []
        for i in val.split("\n"):
            try:
                res.append([datetime.strptime(i[:len(datetime.now().strftime(format))], format), (i[len(datetime.now().strftime(format))+1:] if i[len(datetime.now().strftime(format))] == " " else i[len(datetime.now().strftime(format))+0:])])
            except ValueError:
                res[-1][1] += "\n" + i
        return res

    s1 = make_list(s1) if s1 else []
    s2 = make_list(s2) if s2 else []
    return "\n".join([i[1] for i in sorted(s1 + s2, key=lambda x: x[0])])

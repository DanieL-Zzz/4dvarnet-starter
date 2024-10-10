import collections
import json
import sys

import pandas as pd

if __name__ == "__main__":
    pd.options.display.float_format = "{:,.2f}".format
    d = json.loads(sys.stdin.read())
    out = collections.defaultdict(dict)

    for k, m in d[""]["data"].items():
        method = k.partition("_")[2].partition(".")[0]
        for metric_name, metric_value in m["data"].items():
            out[method][metric_name] = metric_value

    df = pd.DataFrame(out)
    print(df.T.to_markdown(floatfmt=".3f"))

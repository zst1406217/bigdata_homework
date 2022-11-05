import pandas as pd
import pathlib
# dir = pathlib.Path('data')
mypath = "./data"
data = []
folder = pathlib.Path.cwd().joinpath(mypath, 'train')  # 文件夹路径
for fp in folder.iterdir():
    df = pd.read_csv(fp)
    data.append((df, fp))

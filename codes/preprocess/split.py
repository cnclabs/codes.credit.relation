import pandas as pd 
import datetime
import sys

# Format year/month
ys = sys.argv[1]
ms = sys.argv[2]
ye = sys.argv[3]
me = sys.argv[4]

yms = ys + '-' + ms
yme = ye + '-' + me

print("Creating file: Start -- {} ~ End -- {}".format(yms, yme))

data = pd.read_csv(path)

start_date = datetime.datetime.strptime(yms, "%Y-%m").date()
end_date = datetime.datetime.strptime(yme, "%Y-%m").date()

mask = (pd.to_datetime(data['date']) >= start_date) & (pd.to_datetime(data['date']) <= end_date)

df_time = data.loc[mask]
df_time.to_csv(, index=False)

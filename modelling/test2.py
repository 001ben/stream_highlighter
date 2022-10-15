import pandas as pd
df = pd.DataFrame({'id': [1,1], 'day': ['Mon', 'Tue'], 'times': ["4a-4a", "12p-8p"]})

df
df['times'].str.split('-')
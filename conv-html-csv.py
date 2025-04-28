import pandas as pd

# Read the HTML table from the file
with open('paste.txt', 'r', encoding='utf-8') as f:
    html = f.read()

# Parse the table(s) from the HTML
dfs = pd.read_html(html)

# If there is only one table, use the first DataFrame
df = dfs[0]

# Save to CSV
df.to_csv('output.csv', index=False)


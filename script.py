import pandas as pd

df = pd.DataFrame({"x": [10,3,1], "y": [12,4,2]})
# def generate_check(df, high_vector, low_vector):
#     columns = df.columns
#     result = []
   
#     for column, h_vector, l_vector in zip(columns, high_vector, low_vector):
#         calc = ((df[column] <= h_vector) & (df[column] >= l_vector))
#         result.append(set(calc[~calc].index))
        
#     indexes = list(set.union(*result))
#     return indexes

# generate_check(df, [7,5], [1,2])

high_vector = [7, 5]
low_vector = [1,2]

def compare(*args):
    values = list(args)
    if high_vector >= values <= low_vector:
        return tuple(values)

indexes = [] 
for index, row in df.iterrows():
    if high_vector <= list(row) >= low_vector:
        indexes.append(index)
print(indexes)
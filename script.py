import pandas as pd

df = pd.DataFrame({"x": [10,3,1], "y": [12,4,2]})
def generate_check(df, high_vector, low_vector):
    columns = df.columns
    result = []
   
    for column, h_vector, l_vector in zip(columns, high_vector, low_vector):
        calc = ((df[column] <= h_vector) & (df[column] >= l_vector))
        result.append(set(calc[~calc].index))
        
    indexes = list(set.union(*result))
    return indexes
    print(indexes)

generate_check(df, [7,5], [1,2])
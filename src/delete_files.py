import glob, os

num_dimentions = 2 # 8 or 2
folder = 'tabular' # plots or tabular
stream_number = 3 # 12 or 3
file_extention = 'csv' # png or csv

for stream_num in range(stream_number):
    for size in [3, 5, 10, 100, 500, 1000]:
        path = os.path.join(os.path.dirname(__file__), '..', 'results', 'synthetic', f'{num_dimentions}-dim', f'{folder}', f'stream {stream_num}', f'{size}', f'*.{file_extention}')
        for f in glob.glob(path):
            print(f)
            os.remove(f)
import glob, os

num_dimentions = 2 # 8 or 2
folder = 'plots' # plots or tabular
stream_number = 3 # 12 or 3
file_extention = 'png' # png or csv

for stream_num in range(stream_number):
    for size in [3, 4, 6, 8, 12, 24]:
        # path = os.path.join(os.path.dirname(__file__), '..', 'results', 'synthetic', f'{num_dimentions}-dim', f'{folder}', f'stream {stream_num}', f'{size}', f'*.{file_extention}')
        # for f in glob.glob(path):
        #     print(f)
        #     os.remove(f)
        path = os.path.join(os.path.dirname(__file__), '..', 'results', 's1', f'{folder}', f'{size}', f'*.{file_extention}')
        for f in glob.glob(path):
            print(f)
            os.remove(f)
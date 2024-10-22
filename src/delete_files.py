import glob, os

num_dimentions = 2 # 8 or 2
folder = 'tabular' # plots or tabular
stream_number = 3 # 12 or 3
file_extention = 'csv' # png or csv
type = 'original' # continuous or original or continuous_previous or original_previous, gas, elec, water, weather

# for stream_num in range(stream_number):
for size in [30,32,48]:
    # path = os.path.join(os.path.dirname(__file__), '..', 'results', 'synthetic', f'{num_dimentions}-dim', f'{folder}', f'stream {stream_num}', f'{size}', f'*.{file_extention}')
    # for f in glob.glob(path):
    #     print(f)
    #     os.remove(f)
    path = os.path.join(os.path.dirname(__file__), '..', 'results', 'kddcup', f'{folder}', f'{size}', f'*.{file_extention}')
    
    # path = os.path.join(os.path.dirname(__file__), '..', 'results', 'ampds', f'{type}', f'{size}', f'{folder}', f'*.{file_extention}')
    for f in glob.glob(path):
        print(f)
        os.remove(f)
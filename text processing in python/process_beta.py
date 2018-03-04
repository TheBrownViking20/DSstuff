import os
import glob

for i in ['fantasy','scifi','horror','mystery_or_detective']:
    os.chdir('F:/Machine Learning/DSstuff/text processing in python/{}'.format(i))
    files = glob.glob('F:/Machine Learning/DSstuff/text processing in python/{}/*.txt'.format(i))
    print(i, end="---->")
    print(files)
    print("\n")
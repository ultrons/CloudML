import os, argparse
from glob import glob
from pprint import pprint
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument(
        '--gs_bucket',
        help = 'path to the cloud storage bucket where the data will reside',
        default = 'gsp-demo-acme'
        )
parser.add_argument(
        '--source',
        help = 'path to the source data from where the split will happen',
        required = True
        )
parser.add_argument(
        '--minority_fraction',
        help = 'fraction to impose on minority class. Default is 0, which means ratio is as is',
        default = 0,
        type = float
        )
parser.add_argument(
        '--minority_oversampling',
        help = 'factor by which minority class will be oversampled,default value used will inverse of minority fraction.',
        default = 0,
        type = int
        )
parser.add_argument(
        '--source_fraction',
        help = 'fraction of source dataset to be used in generating the split',
        default = 1,
        type = float
        )
parser.add_argument(
        '--tdt_split',
        help = 'train-dev-test split',
        nargs = '+',
        default = [80, 10, 10],
        type = int
        )
args = parser.parse_args()
arguments = args.__dict__
#print(args.source_fraction)

# Get all the data and shuffle
cat=glob(args.source+"/cat*.jpg")
dog=glob(args.source+"/dog*.jpg")
shuffle(cat)
shuffle(dog)

# get target number for majority class
# and set minority class numbers accordingly
dog = dog[0:int(args.source_fraction * len(dog))]
if args.minority_fraction == 0: 
    cat = cat[0:int(args.source_fraction * len(cat))]
else:
    cat = cat[0:int(len(dog)*args.minority_fraction)]


catData = {}
dogData = {}

train, dev, test = [int(0.01*i*len(cat)) for i in args.tdt_split]
# If over sampling factor is specified it will be applied
# if not then it will be invers of minority fraction if minority_fraction is specified
# else no oversampling will be done

if args.minority_oversampling != 0:
    ovr_sampling_factor=args.minority_oversampling
elif args.minority_fraction != 0:
    ovr_sampling_factor = int(1/args.minority_fraction)
else:
    ovr_sampling_factor = 1

catData['train'] = cat[0:train]*ovr_sampling_factor
catData['dev'] = cat[train:train+dev]
catData['test'] = cat[train+dev:]

train, dev, test = [int(0.01*i*len(dog)) for i in args.tdt_split]
dogData['train'] = dog[0:train]
dogData['dev'] = dog[train:train+dev]
dogData['test'] = dog[train+dev:]


for i in ['train', 'dev', 'test']: 
    gs_list=i+'_gs.csv'
    gs=open(gs_list, 'w+')
    allFiles=catData[i]+dogData[i]
    shuffle(allFiles)
    for afile in allFiles:
        if os.path.basename(afile).startswith('cat'):
            label='CAT'
        else:
            label='DOG'
        gs.write("gs://%s/%s,%s\n" %(args.gs_bucket, afile, label))
    gs.close()    


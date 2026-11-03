import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fraudsense.data import load_dataset,inspect_dataset,partitions
parser=argparse.ArgumentParser();parser.add_argument('--data',required=True)
if __name__=='__main__':
    args=parser.parse_args();d=load_dataset(args.data)
    print(json.dumps({'dataset':inspect_dataset(d,args.data),'split':partitions(d)[1]},indent=2))

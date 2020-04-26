import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--choice', type=int)
    args = parser.parse_args()

    if args.choice == 0:
        for fi in range(1, 4):
            for mi in range(1, 6):
                cmd = "python3 main_api_faceplusplus.py --gen 1 --src m%d --tgt f%d"%(mi, fi)
                print(cmd)
                os.system(cmd)
    if args.choice == 1:
        for fi in range(1, 4):
            for mi in range(1, 6):
                cmd = "python3 main_api_faceplusplus.py --gen 1 --src f%d --tgt m%d"%(fi, mi)
                print(cmd)
                os.system(cmd)

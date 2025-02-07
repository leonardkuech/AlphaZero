import pandas as pd

def main():
    branch_df = pd.read_csv('src/data/average_branching.csv')

    print(branch_df[0:100].mean())

if __name__ == '__main__':
    main()
import pandas as pd

def toFloat(s):
    # 数字と','からなる文字列をfloatにする
    ans = ""
    for e in s:
        # ','を無視
        if e != ',':
            ans += e
    return float(ans)

def main():
    df = pd.read_csv('zuhyo01-05-10.csv')

    MAX = 0.0
    ans_str = ""

    # 各要素抽出
    for field, w_str, sum_str in zip(df['分野'], df['女性'], df['総数']):
        w = toFloat(w_str)
        sum = toFloat(sum_str)
        if MAX * sum < w:
            MAX = w / sum
            ans_str = field

    # 出力
    print(f"分野: {ans_str}")
    print(f"割合: {MAX}")

if __name__ == "__main__":
    main()

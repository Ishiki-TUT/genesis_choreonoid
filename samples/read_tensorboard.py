from tensorboard.backend.event_processing import event_accumulator
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="TensorBoard scalar extractor")
    parser.add_argument("logfile", type=str, help="Path to TensorBoard event file")
    parser.add_argument("--out", type=str, default="tensorboard_scalars.png",
                        help="Output PNG file name")
    args = parser.parse_args()

    # TensorBoardのイベントファイルを読み込み
    ea = event_accumulator.EventAccumulator(args.logfile)
    ea.Reload()

    # スカラータグの一覧を取得
    scalar_tags = ea.Tags()['scalars']

    # Episode を含むタグだけを対象にする
    selected_tags = [tag for tag in scalar_tags if 'Episode' in tag]

    data_dict = {}

    # 各タグの step と value を取得
    for tag in selected_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data_dict['step'] = steps  # すべて同じ step を仮定
        data_dict[tag] = values

    # DataFrame にまとめる
    df = pd.DataFrame(data_dict)

    # print(df.head())
    # グラフ描画
    plt.figure(figsize=(10, 6))
    for tag in selected_tags:
        plt.plot(df['step'], df[tag], label=tag)
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('TensorBoard Scalars')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    out_path = args.out
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {os.path.abspath(out_path)}")
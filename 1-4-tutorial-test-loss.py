import torch
import torch.optim as optim
import time
import random
import numpy as np
import matplotlib.pyplot as plt

# 1. 乱数のシードを固定する関数
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 2. 訓練を実行して、最終Lossと実行時間を返す関数
def train_model_quick(device_name, seed):
    set_seed(seed) # 試行ごとに異なるシード、しかしCPU/GPU間では同じシード
    device = torch.device(device_name)

    test_net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(test_net.parameters(), lr=0.001, momentum=0.9)

    start_time = time.time()
    final_loss = 0.0

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = test_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # 2000ミニバッチごとの出力（ノートブック元と同じ設定）
            if i % 2000 == 1999:
                final_loss = running_loss / 2000
                running_loss = 0.0

    exec_time = time.time() - start_time
    return final_loss, exec_time

# 3. 10回試行するメイン処理
num_trials = 10
cpu_times = []
gpu_times = []
loss_diffs = [] # CPUとGPUの最終Lossのズレ（絶対値）
# ★新しくLossそのものを保持するためのリストを追加
cpu_losses = []
gpu_losses = []

if torch.cuda.is_available():
    print(f"{num_trials}回の試行を開始します...")
    for trial in range(num_trials):
        seed = 42 + trial
        print(f"Trial {trial+1}/{num_trials} (実行中...) ", end="")

        # CPUで実行
        cpu_loss, cpu_time = train_model_quick("cpu", seed)
        cpu_times.append(cpu_time)
        cpu_losses.append(cpu_loss)  # ★ここを追加（CPUのLossを保存）

        # GPUで実行
        gpu_loss, gpu_time = train_model_quick("cuda:0", seed)
        gpu_times.append(gpu_time)
        gpu_losses.append(gpu_loss)  # ★ここを追加（GPUのLossを保存）

        # Lossの差を記録
        diff = abs(cpu_loss - gpu_loss)
        loss_diffs.append(diff)

        print(f"完了 (CPU Loss: {cpu_loss:.4f}, GPU Loss: {gpu_loss:.4f}, 誤差: {diff:.6f})")

    # 全て終わった後に各試行のLossを出力して確認する例
    print("\n=== 各試行のLoss一覧 ===")

    for i in range(num_trials):
          print(f"Trial {i+1} - CPU: {cpu_losses[i]:.5f} | GPU: {gpu_losses[i]:.5f}")

    # 4. 結果の集計と可視化
    print("\n" + "="*30)
    print("         結果の集計")
    print("="*30)
    print(f"CPU 平均処理時間: {np.mean(cpu_times):.2f} 秒 (±{np.std(cpu_times):.2f} 秒)")
    print(f"GPU 平均処理時間: {np.mean(gpu_times):.2f} 秒 (±{np.std(gpu_times):.2f} 秒)")
    print(f"最終Lossの計算誤差(平均): {np.mean(loss_diffs):.8f}")

    # 処理時間の棒グラフを描画
    plt.figure(figsize=(10, 5))
    x = np.arange(num_trials)
    width = 0.35

    plt.bar(x - width/2, cpu_times, width, label='CPU Time', color='#1f77b4', alpha=0.8)
    plt.bar(x + width/2, gpu_times, width, label='GPU Time', color='#ff7f0e', alpha=0.8)

    plt.title('Execution Time Comparison (10 Trials)')
    plt.xlabel('Trial Number')
    plt.ylabel('Time (seconds)')
    plt.xticks(x, [f'T{i+1}' for i in range(num_trials)])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

else:
    print("GPUが利用できない環境です。Colabメニューの「ランタイム」>「ランタイムのタイプを変更」からGPUを選択してください。")



import matplotlib.pyplot as plt
import numpy as np

# 保存されているデータの数を取得
num_trials = len(cpu_losses)
x = np.arange(num_trials)

# グラフを横に2つ並べて表示するための設定
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# --- 1つ目のグラフ：CPUとGPUそれぞれのLoss値 ---
# ※値の差が小さすぎるため、線がほぼ完全に重なって表示されます
ax1.plot(x, cpu_losses, marker='o', label='CPU Loss', color='#1f77b4', linewidth=2)
ax1.plot(x, gpu_losses, marker='x', label='GPU Loss', color='#ff7f0e', linewidth=2, linestyle='--')
ax1.set_title('Final Loss: CPU vs GPU (10 Trials)')
ax1.set_xlabel('Trial Number')
ax1.set_ylabel('Loss Value')
ax1.set_xticks(x)
ax1.set_xticklabels([f'T{i+1}' for i in range(num_trials)])
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# --- 2つ目のグラフ：CPUとGPUのLossの差分（絶対誤差） ---
# 変数 loss_diffs がない場合は再計算する
if 'loss_diffs' not in globals() or len(loss_diffs) != num_trials:
    loss_diffs = [abs(c - g) for c, g in zip(cpu_losses, gpu_losses)]

ax2.bar(x, loss_diffs, color='#d62728', alpha=0.7)
ax2.set_title('Absolute Difference (CPU Loss - GPU Loss)')
ax2.set_xlabel('Trial Number')
ax2.set_ylabel('Absolute Error Margin')
ax2.set_xticks(x)
ax2.set_xticklabels([f'T{i+1}' for i in range(num_trials)])
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# 描画
plt.tight_layout()
plt.show()
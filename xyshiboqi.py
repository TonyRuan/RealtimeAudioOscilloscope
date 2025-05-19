import numpy as np
import pyaudio
import tkinter as tk
from tkinter import ttk
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget

# 设置参数
FORMAT = pyaudio.paInt16
CHANNELS = 2  # 立体声
RATE = 44100  # 采样率
CHUNK = 2048  # 每次读取的帧数
DISPLAY_POINTS = 1200  # 显示的点数
sample_step = 1  # 采样步长，每隔几个点取一个点

# 初始化音频流
p = pyaudio.PyAudio()

# 获取所有可用的音频设备
audio_devices = []
stereo_mix_device = None  # 新增变量
print("可用的音频设备:")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    print(f"设备 {i}: {dev['name']}")
    print(f"  输入通道: {dev['maxInputChannels']}")
    print(f"  输出通道: {dev['maxOutputChannels']}")
    print(f"  默认采样率: {dev['defaultSampleRate']}")
    
    # 只添加有输入通道的设备
    if dev['maxInputChannels'] > 0:
        audio_devices.append((i, dev['name'], dev['maxInputChannels'], int(dev['defaultSampleRate'])))
        # 优先精确匹配“立体声混音 (Realtek(R) Audio)”
        if dev['name'] == '立体声混音 (Realtek(R) Audio)':
            stereo_mix_device = (i, dev['name'], dev['maxInputChannels'], int(dev['defaultSampleRate']))
# 如果没有精确匹配，再模糊匹配
if not stereo_mix_device:
    for device in audio_devices:
        if ('立体声混音' in device[1]) or ('Stereo Mix' in device[1]):
            stereo_mix_device = device
            break

# 创建设备选择窗口
def select_audio_device():
    device_window = tk.Tk()
    device_window.title("选择音频设备")
    device_window.geometry("600x400")
    
    # 添加说明标签
    label = tk.Label(device_window, text="请选择音频输入设备（推荐选择'立体声混音'或'Stereo Mix'等设备以捕获系统声音）", wraplength=550)
    label.pack(pady=10)
    
    # 创建设备列表框
    frame = tk.Frame(device_window)
    frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    # 创建表格
    columns = ('设备ID', '设备名称', '输入通道', '采样率')
    tree = ttk.Treeview(frame, columns=columns, show='headings')
    
    # 定义列
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)
    
    # 设置列宽
    tree.column('设备名称', width=250)
    
    # 添加设备到列表
    for device in audio_devices:
        tree.insert('', tk.END, values=device)
    
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # 添加滚动条
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # 选择结果
    selected_device = [None]
    
    # 确认按钮回调
    def on_confirm():
        selection = tree.selection()
        if selection:
            item = tree.item(selection[0])
            selected_device[0] = item['values']
            device_window.destroy()
    
    # 添加确认按钮
    confirm_button = tk.Button(device_window, text="确认", command=on_confirm)
    confirm_button.pack(pady=10)
    
    # 运行窗口
    device_window.mainloop()
    
    return selected_device[0]

# 让用户选择设备
# 自动选择“立体声混音”设备，否则弹窗
if stereo_mix_device:
    selected_device_info = stereo_mix_device
    print(f"自动选择了立体声混音设备: {stereo_mix_device[1]}")
else:
    selected_device_info = select_audio_device()

# 如果用户选择了设备
if selected_device_info:
    input_device_index = selected_device_info[0]
    device_channels = selected_device_info[2]
    device_rate = selected_device_info[3]
    actual_channels = min(CHANNELS, device_channels)
    
    print(f"用户选择设备: {selected_device_info[1]}")
    print(f"使用通道数: {actual_channels}")
    print(f"使用采样率: {device_rate}")
    
    # 尝试打开选定的设备
    stream = None
    success = False
    
    try:
        print(f"尝试打开设备: index={input_device_index}, channels={actual_channels}, rate={device_rate}")
        stream = p.open(format=FORMAT,
                      channels=actual_channels,
                      rate=device_rate,
                      input=True,
                      input_device_index=input_device_index,
                      frames_per_buffer=CHUNK)
        test_data = stream.read(CHUNK, exception_on_overflow=False)
        success = True
        print(f"成功打开设备: {selected_device_info[1]}")
    except Exception as e:
        print(f"打开设备失败: {e}")
        import traceback
        traceback.print_exc()
        if stream:
            stream.close()
            stream = None
        # 打开失败，进入手动选择
        selected_device_info = select_audio_device()
        if selected_device_info:
            input_device_index = selected_device_info[0]
            device_channels = selected_device_info[2]
            device_rate = selected_device_info[3]
            actual_channels = min(CHANNELS, device_channels)
            try:
                print(f"再次尝试打开设备: index={input_device_index}, channels={actual_channels}, rate={device_rate}")
                stream = p.open(format=FORMAT,
                              channels=actual_channels,
                              rate=device_rate,
                              input=True,
                              input_device_index=input_device_index,
                              frames_per_buffer=CHUNK)
                test_data = stream.read(CHUNK, exception_on_overflow=False)
                success = True
                print(f"成功打开设备: {selected_device_info[1]}")
            except Exception as e:
                print(f"再次打开设备失败: {e}")
                if stream:
                    stream.close()
                    stream = None
else:
    # 用户没有选择设备，使用模拟数据
    success = False
    actual_channels = 2
    print("未选择设备，将使用模拟数据")

# 如果设备打开失败，使用模拟数据
if not success:
    print("将使用模拟数据")
    actual_channels = 2
    
    # 模拟音频数据生成函数
    def generate_audio_data():
        t = time.time() * 5
        # 生成利萨如图形
        x = np.sin(t * 2) * 10000
        y = np.sin(t * 3) * 10000
        return x, y


app = QtWidgets.QApplication([])

# 创建PyQtGraph窗口
win = pg.GraphicsLayoutWidget()
win.setWindowTitle('实时音频示波器' + (" (模拟数据)" if not success else ""))
win.resize(800, 800)
win.setBackground('black')

# 创建绘图区域
plot = win.addPlot()
plot.hideAxis('left')
plot.hideAxis('bottom')
plot.setRange(xRange=[-12000, 12000], yRange=[-12000, 12000])  # 范围缩小，更贴合数据

# 创建散点图（只保留一个，去除辉光和渐变）
scatter = pg.ScatterPlotItem(size=3, pen=pg.mkPen('#00ff00', width=0.5), brush=pg.mkBrush('#00ff00'))
plot.addItem(scatter)

# 存储数据点
x_data = np.zeros(DISPLAY_POINTS * 2)
y_data = np.zeros(DISPLAY_POINTS * 2)

# 新增：创建累积图像
accumulate_size = 512  # 图像分辨率，可调整
accumulate_img = np.zeros((accumulate_size, accumulate_size), dtype=np.float32)

# 新增：创建ImageItem用于显示
img_item = pg.ImageItem()
rect_set = False  # 新增标志变量
# 设置ImageItem的坐标映射，使其覆盖整个plot区域

plot.addItem(img_item)
scatter.setVisible(False)  # 隐藏原有散点

# 在第一次 setImage 后设置 rect
img_item.setImage(accumulate_img, levels=(0, 7), autoLevels=False)


# 更新函数
def update():
    global x_data, y_data, accumulate_img, rect_set

    try:
        if success:
            data = stream.read(CHUNK//2, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            if actual_channels == 2:
                left = audio_data[0::2*sample_step].astype(np.int32)
                right = audio_data[1::2*sample_step].astype(np.int32)
            else:
                left = audio_data[::sample_step].astype(np.int32)
                right = audio_data[::sample_step].astype(np.int32)
        else:
            t = int(time.time() * 5) % 360
            left_val = np.sin(np.radians(t * 2)) * 10000
            right_val = np.sin(np.radians(t * 3)) * 10000
            left = np.full(CHUNK//(2*sample_step), left_val, dtype=np.int32)
            right = np.full(CHUNK//(2*sample_step), right_val, dtype=np.int32)

        max_points = min(2000, len(left))

        # 正确的左右镜像：x轴取反
        current_x = -right[:max_points]  # 左右镜像
        current_y = -left[:max_points]   # 上下镜像（新增：取反实现左右+上下镜像）
        points_to_add = len(current_x)
        x_data = np.roll(x_data, points_to_add)
        y_data = np.roll(y_data, points_to_add)
        x_data[:points_to_add] = current_x
        y_data[:points_to_add] = current_y

        scatter.setData(x=x_data, y=y_data)

        x_idx = ((-right[:max_points] + 32768) / 65536 * (accumulate_size - 1)).astype(int)  # 左右镜像
        y_idx = ((-left[:max_points] + 32768) / 65536 * (accumulate_size - 1)).astype(int)   # 上下镜像（新增：取反实现左右+上下镜像）

        accumulate_img *= 0.8
        for xi, yi in zip(x_idx, y_idx):
            if 0 <= xi < accumulate_size and 0 <= yi < accumulate_size:
                accumulate_img[yi, xi] += 1.0

        img_item.setImage(accumulate_img, levels=(0, 7), autoLevels=False)
        if not rect_set:
            img_item.setRect(QtCore.QRectF(-12000, -12000, 24000, 24000))
            rect_set = True
        print("max:", accumulate_img.max(), "min:", accumulate_img.min())

    except Exception as e:
        print(f"错误: {e}")

# 定义关闭事件处理函数
def on_closing():
    if stream:
        stream.stop_stream()
        stream.close()
    p.terminate()
    app.quit()  # 使用app.quit()而不是root.destroy()

# 不再需要Tkinter的关闭事件处理
# root.protocol("WM_DELETE_WINDOW", on_closing)

# 使用QTimer替代FuncAnimation
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(5)  # 5ms刷新间隔

# 显示窗口
win.show()

# 使用PyQt的主循环，而不是Tkinter的
# tk.mainloop()
app.exec_()  # 使用PyQt的主循环

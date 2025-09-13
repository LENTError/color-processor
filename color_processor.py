import sys
import cv2
import numpy as np
import os
import zipfile
import hashlib
import threading
import time
import logging
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QScrollArea, QGridLayout, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QPixmap, QImage, QIcon
from qfluentwidgets import (
    PushButton, HyperlinkButton, ComboBox, TextEdit, CompactSpinBox, CheckBox,
    InfoBarIcon, InfoBar, InfoBarPosition, MessageBox, ProgressBar,
    FluentIcon as FIF, Flyout, FlyoutView
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resource_path(relative_path):
    if getattr(sys, "_MEIPASS", None):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

icon_path = resource_path("favicon.ico")
res_github = resource_path("res/github.png")
res_jsfmytg = resource_path("res/jsfmytg.jpg")
res_LENTError = resource_path("res/LENTError.png")

class VideoPreviewThread(QThread):
    """视频预览线程，支持跳转到指定帧并暂停"""
    frame_ready = pyqtSignal(QImage, int, float)  # 图像, 帧索引, 时间戳
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.running = False
        self.mutex = QMutex()
        self.current_frame_index = 0
        self.target_frame_index = 0
        self.seek_requested = False
        self.cap = None
        self.paused = True  # 默认暂停状态
        
    def run(self):
        self.mutex.lock()
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.mutex.unlock()
            return
            
        self.running = True
        self.mutex.unlock()
        
        # 初始显示第一帧
        self.seek_to_frame(0)
        
        while self.running:
            self.mutex.lock()
            if not self.running:
                self.mutex.unlock()
                break
                
            # 检查是否有跳转请求
            if self.seek_requested:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.target_frame_index)
                self.current_frame_index = self.target_frame_index
                self.seek_requested = False
                
                # 读取并显示帧
                ret, frame = self.cap.read()
                if ret:
                    # 转换为QImage
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame_rgb.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    
                    # 获取时间戳
                    timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    
                    self.mutex.unlock()
                    self.frame_ready.emit(qt_image, self.current_frame_index, timestamp)
                else:
                    self.mutex.unlock()
                
                continue
            
            self.mutex.unlock()
            time.sleep(0.01)  # 减少CPU占用
            
        if self.cap:
            self.cap.release()
        
    def seek_to_frame(self, frame_index):
        """跳转到指定帧"""
        self.mutex.lock()
        self.target_frame_index = frame_index
        self.seek_requested = True
        self.mutex.unlock()
        
    def stop(self):
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()

class ProcessingThread(QThread):
    """处理线程"""
    progress_updated = pyqtSignal(int, int, str)
    frame_processed = pyqtSignal(int, list, float)  # 帧索引, 颜色列表, 时间戳
    seek_preview = pyqtSignal(int)  # 请求预览跳转
    finished = pyqtSignal(bool, str)
    
    def __init__(self, processor, input_path, output_dir, frame_rate, color_mode, 
                 color_count, manual_colors, max_width, create_zip):
        super().__init__()
        self.processor = processor
        self.input_path = input_path
        self.output_dir = output_dir
        self.frame_rate = frame_rate
        self.color_mode = color_mode
        self.color_count = color_count
        self.manual_colors = manual_colors
        self.max_width = max_width
        self.create_zip = create_zip
        self.stop_flag = False
        self.mutex = QMutex()
        
    def run(self):
        try:
            # 确定文件类型
            file_ext = os.path.splitext(self.input_path)[1].lower()
            is_video = file_ext in ['.mp4', '.mov', '.flv']
            
            if is_video:
                success = self.process_video()
            elif is_image:
                success = self.process_image()
            else:
                self.finished.emit(False, f"不支持的文件格式: {file_ext}")
                return
                
            if success and self.create_zip and not self.stop_flag:
                zip_name = 'video_frames.zip' if is_video else 'image_frame.zip'
                zip_path = os.path.join(self.output_dir, zip_name)
                self.processor.create_zip_archive(self.output_dir, zip_path)
                self.finished.emit(True, f"处理完成！结果已保存到: {self.output_dir}\nZIP文件: {zip_path}")
            elif success:
                self.finished.emit(True, f"处理完成！结果已保存到: {self.output_dir}")
            else:
                self.finished.emit(False, "处理被用户取消")
                
        except Exception as e:
            logger.error(f"处理过程中出错: {e}")
            self.finished.emit(False, f"处理失败: {str(e)}")
    
    def process_video(self):
        """处理视频文件"""
        self.mutex.lock()
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            self.mutex.unlock()
            raise Exception(f"无法打开视频文件: {self.input_path}")
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 确保fps不为0
        if fps <= 0:
            fps = 30
            
        # 计算要提取的帧
        frame_interval = max(1, int(fps / self.frame_rate))
        frame_indices = list(range(0, total_frames, frame_interval))
        total_to_process = len(frame_indices)
        
        self.mutex.unlock()
        
        self.progress_updated.emit(0, total_to_process, f"开始处理视频，共 {total_to_process} 帧")
        
        processed_count = 0
        
        for i, frame_idx in enumerate(frame_indices):
            self.mutex.lock()
            if self.stop_flag:
                self.mutex.unlock()
                break
            self.mutex.unlock()
            
            # 请求预览跳转到当前帧
            self.seek_preview.emit(frame_idx)
            time.sleep(0.05)  # 给预览线程一点时间跳转
            
            # 读取帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"无法读取帧 {frame_idx}")
                continue
                
            # 获取时间戳
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # 转换为秒
            
            # 调整大小 - 确保使用用户设置的最大宽度
            if self.max_width > 0 and frame.shape[1] > self.max_width:
                scale = self.max_width / frame.shape[1]
                new_width = self.max_width
                new_height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # 计算文件夹命名（根据颜色数量分组）
            folder_index = i
            folder_start = folder_index * self.color_count
            folder_end = folder_start + self.color_count - 1
            folder_name = f"{folder_start}_{folder_end}"
            
            # 处理帧
            try:
                result = self.processor.process_frame(
                    frame, i, self.output_dir, folder_name, self.color_mode,
                    self.color_count, self.manual_colors
                )
                if result:
                    colors = result['colors']
                    self.frame_processed.emit(frame_idx, colors, timestamp)
                    processed_count += 1
                    
                    # 更新进度
                    self.progress_updated.emit(processed_count, total_to_process, 
                                              f"处理帧 {processed_count}/{total_to_process} (时间: {timestamp:.2f}s)")
            except Exception as e:
                logger.error(f"处理帧 {i} 时出错: {e}")
        
        cap.release()
        return not self.stop_flag and processed_count > 0
    
    def process_image(self):
        """处理图像文件"""
        self.mutex.lock()
        if self.stop_flag:
            self.mutex.unlock()
            return False
            
        image = cv2.imread(self.input_path)
        if image is None:
            self.mutex.unlock()
            raise Exception(f"无法读取图像文件: {self.input_path}")
        
        # 调整大小 - 确保使用用户设置的最大宽度
        if self.max_width > 0 and image.shape[1] > self.max_width:
            scale = self.max_width / image.shape[1]
            new_width = self.max_width
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        self.mutex.unlock()
        
        self.progress_updated.emit(0, 1, "开始处理图像")
        
        # 计算文件夹命名（根据颜色数量）
        folder_start = 0
        folder_end = self.color_count - 1
        folder_name = f"{folder_start}_{folder_end}"
        
        # 处理图像
        result = self.processor.process_frame(
            image, 0, self.output_dir, folder_name, self.color_mode, 
            self.color_count, self.manual_colors
        )
        
        if result:
            colors = result['colors']
            self.frame_processed.emit(0, colors, 0.0)
        
        self.progress_updated.emit(1, 1, "图像处理完成")
        return not self.stop_flag
    
    def stop(self):
        """停止处理"""
        self.mutex.lock()
        self.stop_flag = True
        self.mutex.unlock()

class FastColorProcessor:
    """快速颜色处理器"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(4, os.cpu_count() or 2)
        
    def parse_manual_colors(self, color_input: str) -> List[Tuple[int, int, int]]:
        """解析手动输入的颜色列表"""
        colors = []
        if not color_input.strip():
            return colors
            
        lines = color_input.strip().split('\n')
        
        for line in lines:
            line = line.strip().upper()
            if not line:
                continue
                
            if line.startswith('#'):
                line = line[1:]
                
            if len(line) == 3:
                try:
                    r = int(line[0] + line[0], 16)
                    g = int(line[1] + line[1], 16)
                    b = int(line[2] + line[2], 16)
                    colors.append((r, g, b))
                except ValueError:
                    continue
            elif len(line) == 6:
                try:
                    r = int(line[0:2], 16)
                    g = int(line[2:4], 16)
                    b = int(line[4:6], 16)
                    colors.append((r, g, b))
                except ValueError:
                    continue
                    
        return colors
    
    def kmeans_quantize(self, image: np.ndarray, color_count: int) -> List[Tuple[int, int, int]]:
        """使用K-means算法进行颜色量化"""
        # 采样部分像素以提高速度
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        if len(pixels) > 5000:
            indices = np.random.choice(len(pixels), 5000, replace=False)
            pixels = pixels[indices]
        
        # 使用K-means聚类
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        _, labels, centers = cv2.kmeans(pixels, color_count, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # 转换为整数颜色
        colors = [tuple(map(int, center)) for center in centers]
        return colors
    
    def generate_color_mask_fast(self, args: Tuple[np.ndarray, List[Tuple[int, int, int]], int, str]) -> Tuple[int, np.ndarray]:
        """快速生成颜色掩码"""
        image, colors, color_idx, output_path = args
        target_color = np.array(colors[color_idx])
        
        # 使用向量化操作计算距离
        distances = np.sqrt(np.sum((image - target_color) ** 2, axis=2))
        
        # 找到最小距离的索引
        all_distances = np.stack([np.sqrt(np.sum((image - np.array(color)) ** 2, axis=2)) 
                                for color in colors], axis=-1)
        min_indices = np.argmin(all_distances, axis=-1)
        
        # 创建掩码
        mask = (min_indices == color_idx).astype(np.uint8) * 255
        
        # 保存掩码图像
        if output_path:
            cv2.imwrite(output_path, mask)
        
        return color_idx, mask
    
    def process_frame(self, frame: np.ndarray, frame_idx: int, output_dir: str, folder_name: str,
                     color_mode: str, color_count: int, manual_colors: str = None) -> Dict[str, Any]:
        """处理单个帧"""
        try:
            if color_mode == 'auto':
                colors = self.kmeans_quantize(frame, color_count)
            else:
                colors = self.parse_manual_colors(manual_colors)
                if not colors:
                    raise ValueError("手动颜色解析失败，请提供有效的颜色值")
            
            # 创建帧目录（使用新的命名规范）
            frame_dir = os.path.join(output_dir, folder_name)
            os.makedirs(frame_dir, exist_ok=True)
            
            # 准备处理参数
            processing_args = []
            for color_idx, color in enumerate(colors):
                hex_color = f"{color[0]:02X}{color[1]:02X}{color[2]:02X}"
                # 文件名格式：颜色值.png
                output_path = os.path.join(frame_dir, f"{hex_color}.png")
                processing_args.append((frame, colors, color_idx, output_path))
            
            # 并行处理颜色掩码
            results = {}
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(processing_args))) as executor:
                futures = {
                    executor.submit(self.generate_color_mask_fast, args): args[2] 
                    for args in processing_args
                }
                
                for future in as_completed(futures):
                    color_idx = futures[future]
                    try:
                        result_color_idx, mask = future.result(timeout=30)
                        results[result_color_idx] = {
                            'color': colors[result_color_idx],
                            'mask': mask
                        }
                    except Exception as e:
                        logger.error(f"处理颜色 {color_idx} 时出错: {e}")
            
            return {
                'frame_idx': frame_idx,
                'folder_name': folder_name,
                'colors': colors,
                'masks': results
            }
            
        except Exception as e:
            logger.error(f"处理帧 {frame_idx} 时出错: {e}")
            return None
    
    def create_zip_archive(self, source_dir: str, output_zip: str) -> bool:
        """创建ZIP压缩包"""
        try:
            with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(source_dir):
                    for file in files:
                        if file == os.path.basename(output_zip):
                            continue
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, source_dir)
                        zipf.write(file_path, arcname)
            
            logger.info(f"ZIP压缩包创建成功: {output_zip}")
            return True
        except Exception as e:
            logger.error(f"创建ZIP压缩包时出错: {e}")
            return False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.processor = FastColorProcessor()
        self.input_path = ""
        self.output_dir = ""
        self.preview_thread = None
        self.processing_thread = None
        self.current_frame_index = -1
        self.total_frames = 0
        self.is_processing = False
        self.init_ui()
    def init_ui(self):
        self.setWindowTitle(f"视频帧颜色分割处理工具")
        self.setWindowIcon(QIcon(icon_path))
        self.setGeometry(100, 100, 1060, 800)
        self.setFixedSize(1060, 800)
        
        # 创建主控件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # 左侧配置面板
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        
        # 右侧预览面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 配置左侧面板
        self.setup_left_panel(left_layout)
        
        # 配置右侧面板
        self.setup_right_panel(right_layout)
        
        # 添加左右面板
        layout.addWidget(left_panel)
        layout.addWidget(right_panel, 1)

        
    def setup_left_panel(self, layout):
        """设置左侧配置面板"""
        # 文件选择区域
        file_group = QGroupBox("文件选择")
        file_layout = QVBoxLayout(file_group)
        
        input_layout = QHBoxLayout()
        self.input_label = QLabel("未选择文件")
        self.input_label.setWordWrap(True)
        input_btn = PushButton(FIF.VIDEO, '选择文件')
        input_btn.clicked.connect(self.select_input_file)
        input_layout.addWidget(self.input_label, 3)
        input_layout.addWidget(input_btn, 1)
        file_layout.addLayout(input_layout)
        
        output_layout = QHBoxLayout()
        self.output_label = QLabel("未选择输出目录")
        self.output_label.setWordWrap(True)
        output_btn = PushButton(FIF.FOLDER, '输出目录')
        output_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(self.output_label, 3)
        output_layout.addWidget(output_btn, 1)
        file_layout.addLayout(output_layout)
        
        layout.addWidget(file_group)
        
        # 处理参数区域
        params_group = QGroupBox("处理参数")
        params_layout = QVBoxLayout(params_group)
        
        # 帧率设置
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("帧率 (FPS):"))
        self.fps_spin = CompactSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(25)
        fps_layout.addWidget(self.fps_spin)
        fps_layout.addStretch()
        params_layout.addLayout(fps_layout)
        
        # 颜色模式
        color_mode_layout = QHBoxLayout()
        self.color_mode_combo = ComboBox()
        self.color_mode_combo.setPlaceholderText("颜色模式")
        color_mode_options = ['自动', '手动']
        self.color_mode_combo.addItems(color_mode_options)
        self.color_mode_combo.setCurrentIndex(-1)
        self.color_mode_combo.currentTextChanged.connect(self.toggle_color_mode)
        color_mode_layout.addWidget(self.color_mode_combo)
        color_mode_layout.addStretch()
        params_layout.addLayout(color_mode_layout)
        
        # 自动颜色设置
        auto_color_layout = QHBoxLayout()
        auto_color_layout.addWidget(QLabel("颜色数量:"))
        self.color_count_spin = CompactSpinBox()
        self.color_count_spin.setRange(1, 256)
        self.color_count_spin.setValue(32)
        auto_color_layout.addWidget(self.color_count_spin)
        auto_color_layout.addStretch()
        params_layout.addLayout(auto_color_layout)
        
        # 手动颜色设置
        manual_color_layout = QVBoxLayout()
        manual_color_layout.addWidget(QLabel("颜色列表 (每行一个):"))
        self.manual_colors_text = TextEdit()
        self.manual_colors_text.setPlaceholderText("000000\nFFFFFF\nFF0000")
        manual_color_layout.addWidget(self.manual_colors_text)
        params_layout.addLayout(manual_color_layout)
        self.manual_colors_text.hide()
        
        # 最大宽度
        max_width_layout = QHBoxLayout()
        max_width_layout.addWidget(QLabel("最大宽度 (像素):"))
        self.max_width_spin = CompactSpinBox()
        self.max_width_spin.setRange(30, 2000)
        self.max_width_spin.setValue(30)
        max_width_layout.addWidget(self.max_width_spin)
        max_width_layout.addStretch()
        params_layout.addLayout(max_width_layout)
        
        # 创建ZIP选项
        self.zip_checkbox = CheckBox("创建ZIP压缩包")
        self.zip_checkbox.setChecked(True)
        params_layout.addWidget(self.zip_checkbox)
        
        layout.addWidget(params_group)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        self.process_btn = PushButton(FIF.PLAY, '开始处理', self)
        self.process_btn.clicked.connect(self.start_processing)
        self.stop_btn = PushButton(FIF.PAUSE, '停止处理', self)
        self.stop_btn.setToolTip('停止处理后，重启应用以处理新内容⚠️')
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        
        control_layout.addWidget(self.process_btn)
        control_layout.addWidget(self.stop_btn)
        layout.addLayout(control_layout)
        
        # 进度条
        self.progress_bar = ProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("准备就绪")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        # 当前帧信息
        self.frame_info_label = QLabel("当前帧: -")
        layout.addWidget(self.frame_info_label)
        
        layout.addStretch()

        self.vBoxLayout = QHBoxLayout()
        bottom_buttons_layout = QHBoxLayout()
        bottom_buttons_layout.setAlignment(Qt.AlignCenter)  # 居中
        self.introduction_label = QLabel("本软件是在'video-to-bilibili-bas'项目中的'视频拆分颜色.html'工具基础上开发。\n\n")
        self.introduction_label.setWordWrap(True)
        layout.addWidget(self.introduction_label)
        self.github = PushButton(FIF.GITHUB, "Github")
        self.bili1 = PushButton(FIF.PEOPLE, "jsfmytg")
        self.bili2 = PushButton(FIF.PEOPLE, "LENTError")
        bottom_buttons_layout.addWidget(self.github)
        bottom_buttons_layout.addWidget(self.bili1)
        bottom_buttons_layout.addWidget(self.bili2)
        layout.addLayout(bottom_buttons_layout)

        self.github.clicked.connect(self.Github)
        self.bili1.clicked.connect(self.Bili_1)
        self.bili2.clicked.connect(self.Bili_2)

        
    def Bili_1(self):
        view = FlyoutView(
            title='jsfmytg',
            content="UID 3546828647696829\n随缘更新质量不定，绝不接广绝不卖号不开收益",
            image=res_jsfmytg,
            isClosable=True
        )
        self.button = HyperlinkButton(
            url='https://space.bilibili.com/3546828647696829',
            text='主页',
            parent=self,
            icon=FIF.LINK
        )
        view.addWidget(self.button, align=Qt.AlignRight)
        view.widgetLayout.insertSpacing(1, 5)
        view.widgetLayout.addSpacing(5)
        w = Flyout.make(view, self.bili1, self)
        view.closed.connect(w.close)

    def Bili_2(self):
        view = FlyoutView(
            title='LENTError',
            content="UID 5560851\n我欲触碰苍穹，却被无形的枷锁紧紧束缚，\n心中仍燃烧着不灭的渴望",
            image=res_LENTError,
            isClosable=True
        )
        self.button = HyperlinkButton(
            url='https://space.bilibili.com/5560851',
            text='主页',
            parent=self,
            icon=FIF.LINK
        )
        view.addWidget(self.button, align=Qt.AlignRight)
        view.widgetLayout.insertSpacing(1, 5)
        view.widgetLayout.addSpacing(5)
        w = Flyout.make(view, self.bili1, self)
        view.closed.connect(w.close)

    def Github(self):
        view = FlyoutView(
            title='相关链接',
            content="",
            image=res_github,
            isClosable=True
        )
        self.button = HyperlinkButton(
            url='https://github.com/bszapp/video-to-bilibili-bas',
            text='一个将视频转换成bilibili高级弹幕工具',
            parent=self,
            icon=FIF.LINK
        )
        view.addWidget(self.button, align=Qt.AlignRight)

        self.button2 = HyperlinkButton(
            url='https://github.com/LENTError/',
            text='视频帧颜色分割处理工具',
            parent=self,
            icon=FIF.LINK
        )
        view.addWidget(self.button2, align=Qt.AlignRight)

        # adjust layout (optional)
        view.widgetLayout.insertSpacing(1, 5)
        view.widgetLayout.addSpacing(5)

        # show view
        w = Flyout.make(view, self.bili1, self)
        view.closed.connect(w.close)


    
    def setup_right_panel(self, layout):
        """设置右侧预览面板"""
        # 原始视频预览
        preview_group = QGroupBox("原始视频预览")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedSize(640, 360)
        self.preview_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.preview_label.setText("无预览")
        preview_layout.addWidget(self.preview_label)
        
        # 视频信息
        self.video_info_label = QLabel("视频信息: 未加载")
        preview_layout.addWidget(self.video_info_label)
        
        layout.addWidget(preview_group)
        
        # 颜色预览
        colors_group = QGroupBox("检测到的颜色")
        colors_layout = QVBoxLayout(colors_group)
        
        self.colors_scroll = QScrollArea()
        self.colors_widget = QWidget()
        self.colors_layout = QGridLayout(self.colors_widget)
        self.colors_scroll.setWidget(self.colors_widget)
        self.colors_scroll.setWidgetResizable(True)
        self.colors_scroll.setMinimumHeight(200)
        colors_layout.addWidget(self.colors_scroll)
        
        layout.addWidget(colors_group)
        
        # 结果标签
        self.result_label = QLabel("尚未处理任何文件")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)
    
    def toggle_color_mode(self, mode):
        if mode == "自动":
            self.color_count_spin.setEnabled(True)
            self.manual_colors_text.hide()
        else:
            self.color_count_spin.setEnabled(False)
            self.manual_colors_text.show()
    
    def select_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频或图像文件", "",
            "媒体文件 (*.mp4 *.mov *.flv)"
        )
        if file_path:
            self.input_path = file_path
            self.input_label.setText(os.path.basename(file_path))
            
            # 停止之前的预览和处理
            self.stop_preview()
            self.stop_processing()
            
            # 启动预览（暂停在第一帧）
            self.start_preview()
    
    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_dir = dir_path
            self.output_label.setText(dir_path)
    
    def start_preview(self):
        if not self.input_path or self.is_processing:
            return
            
        file_ext = os.path.splitext(self.input_path)[1].lower()
        if file_ext in ['.mp4', '.mov', '.flv']:
            # 获取视频信息
            cap = cv2.VideoCapture(self.input_path)
            if cap.isOpened():
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = self.total_frames / fps if fps > 0 else 0
                cap.release()
                
                self.video_info_label.setText(f"总帧数: {self.total_frames}, FPS: {fps:.2f}, 时长: {duration:.2f}秒")
            
            # 启动预览线程（默认暂停在第一帧）
            self.preview_thread = VideoPreviewThread(self.input_path)
            self.preview_thread.frame_ready.connect(self.update_preview)
            self.preview_thread.start()
        else:
            # 图像文件预览
            image = cv2.imread(self.input_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = image.shape
                bytes_per_line = ch * w
                qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.update_preview(qt_image, 0, 0.0)
                self.video_info_label.setText(f"图像尺寸: {w}x{h}")
    
    def stop_preview(self):
        """停止预览"""
        if self.preview_thread:
            self.preview_thread.stop()
            self.preview_thread.wait()
            self.preview_thread = None
    
    def update_preview(self, image, frame_index, timestamp):
        """更新预览画面"""
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled_pixmap)
        
        # 更新帧信息
        self.current_frame_index = frame_index
        self.frame_info_label.setText(f"当前帧: {frame_index}/{self.total_frames} (时间: {timestamp:.2f}s)")
    
    def seek_preview_to_frame(self, frame_index):
        """跳转预览到指定帧"""
        if self.preview_thread:
            self.preview_thread.seek_to_frame(frame_index)
    
    def start_processing(self):
        title = '警告'
        input_error = '请先选择输入文件'
        export_error = '请先选择输出目录'
        if not self.input_path:
            i = MessageBox(title, input_error, self)
            i.setClosableOnMaskClicked(True)
            i.setDraggable(True)
            i.exec()
            return
    
        if not self.output_dir:
            e = MessageBox(title, export_error, self)
            e.setClosableOnMaskClicked(True)
            e.setDraggable(True)
            e.exec()
            return
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 获取参数
        frame_rate = self.fps_spin.value()
        color_mode = "auto" if self.color_mode_combo.currentText() == "自动" else "manual"
        color_count = self.color_count_spin.value()
        manual_colors = self.manual_colors_text.toPlainText()
        max_width = self.max_width_spin.value()
        create_zip = self.zip_checkbox.isChecked()
        
        # 设置处理状态
        self.is_processing = True
        
        # 清空颜色显示
        self.clear_colors_display()
        
        # 启动处理线程
        self.processing_thread = ProcessingThread(
            self.processor, self.input_path, self.output_dir, frame_rate,
            color_mode, color_count, manual_colors, max_width, create_zip
        )
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.frame_processed.connect(self.update_colors_display)
        self.processing_thread.seek_preview.connect(self.seek_preview_to_frame)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.start()
        
        # 更新按钮状态
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        start_content = "处理开始..."
        w = InfoBar(
            icon=InfoBarIcon.INFORMATION,
            title='',
            content=start_content,
            orient=Qt.Vertical,
            isClosable=True,
            position=InfoBarPosition.TOP_RIGHT,
            duration=2000,
            parent=self
        )
        w.show()
    
    def stop_processing(self):
        """停止处理"""
        stop_content = "正在停止处理..."
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
        w = InfoBar(
            icon=InfoBarIcon.INFORMATION,
            title='',
            content=stop_content,
            orient=Qt.Vertical,
            isClosable=True,
            position=InfoBarPosition.TOP_RIGHT,
            duration=2000,
            parent=self
        )
        w.show()
    
    def update_progress(self, current, total, message):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(message)
    
    def clear_colors_display(self):
        """清空颜色显示"""
        for i in reversed(range(self.colors_layout.count())): 
            widget = self.colors_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
    
    def update_colors_display(self, frame_idx, colors, timestamp):
        """更新颜色显示"""
        self.current_frame_index = frame_idx
        self.frame_info_label.setText(f"处理帧: {frame_idx} (时间: {timestamp:.2f}s)")
        
        # 清空之前的颜色显示
        self.clear_colors_display()
        
        # 显示新的颜色
        for i, color in enumerate(colors):
            color_widget = QWidget()
            color_layout = QVBoxLayout(color_widget)
            color_layout.setSpacing(2)
            color_layout.setContentsMargins(2, 2, 2, 2)
            
            # 颜色方块
            color_label = QLabel()
            color_label.setFixedSize(50, 50)
            color_label.setStyleSheet(f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); border: 2px solid black; border-radius: 5px;")
            color_layout.addWidget(color_label)
            
            # 颜色值
            hex_color = f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"
            value_label = QLabel(hex_color)
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setStyleSheet("font-size: 10px;")
            color_layout.addWidget(value_label)
            
            # RGB值
            rgb_label = QLabel(f"RGB: {color[0]},{color[1]},{color[2]}")
            rgb_label.setAlignment(Qt.AlignCenter)
            rgb_label.setStyleSheet("font-size: 9px; color: #666;")
            color_layout.addWidget(rgb_label)
            
            # 添加到网格
            row = i // 6
            col = i % 6
            self.colors_layout.addWidget(color_widget, row, col)
        
        # 更新布局
        self.colors_widget.adjustSize()
    
    def processing_finished(self, success, message):
        # 重置处理状态
        self.is_processing = False
        
        # 更新按钮状态
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if success:

            InfoBar.success(
                title='',
                content="处理完成",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )
        else:
            InfoBar.warning(
                title='',
                content="处理失败",
                orient=Qt.Horizontal,
                isClosable=False,
                position=InfoBarPosition.TOP_LEFT,
                duration=2000,
                parent=self
            )
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        self.stop_preview()
        self.stop_processing()
        event.accept()

def main():
    app = QApplication(sys.argv)
    Window = MainWindow()
    Window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import math
import time
from PIL import Image
import pytesseract
import os
import sys

tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tessdata_dir_prefix = r'C:\Program Files\Tesseract-OCR\tessdata'

# 檢查 Tesseract 是否安裝
if not os.path.exists(tesseract_path):
    print(f"錯誤：Tesseract 未在 {tesseract_path} 找到。請確保已安裝 Tesseract。")
    sys.exit(1)

# 檢查語言數據文件是否存在
if not os.path.exists(os.path.join(tessdata_dir_prefix, 'chi_tra.traineddata')):
    print(f"錯誤：繁體中文語言數據文件未在 {tessdata_dir_prefix} 找到。")
    print("請確保已下載並安裝繁體中文語言包。")
    sys.exit(1)

# 設置 Tesseract 配置
pytesseract.pytesseract.tesseract_cmd = tesseract_path
os.environ['TESSDATA_PREFIX'] = tessdata_dir_prefix

def list_available_languages():
    try:
        languages = pytesseract.get_languages(config='')
        print("可用的語言包:")
        for lang in languages:
            print(f"- {lang}")
    except pytesseract.TesseractNotFoundError:
        print("無法獲取語言列表。請確保 Tesseract 已正確安裝。")

def detect_and_click_button(window):
    while True:
        try:
            # 獲取窗口位置和大小
            left, top, width, height = window.left, window.top, window.width, window.height

            # 截取指定窗口的截圖
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            # 轉換為灰度圖
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

            # 二值化處理
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            # 使用OCR識別文字
            text = pytesseract.image_to_string(Image.fromarray(binary), lang='eng')

            # 檢查是否包含"玩遊戲"
            if "MOONBIX" in text:
                print("檢測到'玩遊戲'")
                
                # 尋找黃色按鈕
                hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([30, 255, 255])
                mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
                
                # 查找輪廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # 找到最大的輪廓（假設是按鈕）
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # 計算按鈕中心（相對於窗口）
                    center_x = left + x + w // 2
                    center_y = top + y + h // 2
                    
                    # 點擊按鈕
                    pyautogui.click(center_x, center_y)
                    print(f"點擊了位置：({center_x}, {center_y})")
                    
                    # 連續點擊29秒，每0.1秒一次
                    start_time = time.time()
                    while time.time() - start_time < 29:
                        pyautogui.click(center_x, center_y)
                        time.sleep(0.1)
                    
                    print("完成29秒連續點擊")
                
                # 完成一輪操作後，短暫等待
                time.sleep(1)
            else:
                print("未檢測到開始")
            # 短暫休息，避免過度佔用CPU
            time.sleep(0.5)

        except pytesseract.TesseractError as e:
            print(f"Tesseract 錯誤：{str(e)}")
            print("請確保 Tesseract 和繁體中文語言包已正確安裝。")
            list_available_languages()
            break
        except Exception as e:
            print(f"發生錯誤：{str(e)}")
            break

def click_window_center(window):
    left, top, width, height = window.left, window.top, window.width, window.height
    center_x = left + width // 2
    center_y = top + height * 3/4

    try:
        while True:
            # 模拟点击窗口中心
            pyautogui.click(center_x, center_y)
            print(f"Clicked at ({center_x}, {center_y}) in window '{window.title}'")
            
            # 等待一秒
            time.sleep(1)
    except KeyboardInterrupt:
        print("Clicking stopped.")

def detect_objects_in_window(window):
    while True:
        # 捕获窗口截图
        left, top, width, height = window.left, window.top, window.width, window.height
        screenshot = pyautogui.screenshot(region=(left, top, width, height))
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 金币检测（黄色）
        lower_gold = np.array([20, 100, 100])
        upper_gold = np.array([30, 255, 255])
        gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
        
        # 爪子检测（黄色，但可能需要调整）
        lower_claw = np.array([20, 100, 100])
        upper_claw = np.array([30, 255, 255])
        claw_mask = cv2.inRange(hsv, lower_claw, upper_claw)

        # 隕石检测（灰色到黑色）
        lower_meteor = np.array([0, 0, 0])
        upper_meteor = np.array([180, 50, 100])
        meteor_mask = cv2.inRange(hsv, lower_meteor, upper_meteor)

        # 寻找轮廓
        gold_contours, _ = cv2.findContours(gold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        claw_contours, _ = cv2.findContours(claw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        meteor_contours, _ = cv2.findContours(meteor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 金币检测
        for contour in gold_contours:
            area = cv2.contourArea(contour)
            if 30 < area < 3000:
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.6:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Coin', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 爪子检测
        for contour in claw_contours:
            area = cv2.contourArea(contour)
            if area > 500:  # 爪子通常比金币大
                # 计算最小外接矩形
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype=np.int32)  # 使用 np.int32 替代 np.int0

                # 计算旋转角度
                angle = rect[2]
                if angle < -45:
                    angle += 90

                # 绘制矩形和角度
                cv2.drawContours(frame, [box], 0, (255, 165, 0), 2)
                center = tuple(map(int, rect[0]))
                cv2.putText(frame, f'Claw {angle:.1f}°', (center[0]-40, center[1]-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

                # 绘制方向箭头
                end_point = (int(center[0] + 50 * math.cos(math.radians(angle))),
                             int(center[1] + 50 * math.sin(math.radians(angle))))
                cv2.arrowedLine(frame, center, end_point, (255, 165, 0), 2)

        # 隕石检测
        for contour in meteor_contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                if 0.7 < aspect_ratio < 1.3:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, 'Meteor', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 显示结果
        cv2.imshow(f'Detected Objects in {window.title}', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def list_and_select_window():
    windows = [win for win in gw.getAllWindows() if win.title]
    print("当前打开的窗口:")
    for i, window in enumerate(windows, 1):
        print(f"{i}. {window.title}")
    
    while True:
        try:
            choice = int(input("请选择要检测的窗口编号: "))
            if 1 <= choice <= len(windows):
                return windows[choice - 1]
            else:
                print("无效的选择，请重试。")
        except ValueError:
            print("请输入有效的数字。")


def main():
    print("正在檢查可用的語言包...")
    list_available_languages()
    #click_window_center(selected_window)
    #detect_objects_in_window(selected_window)
    selected_window = list_and_select_window()
    print(f"您選擇了: {selected_window.title}")
    detect_and_click_button(selected_window)

if __name__ == "__main__":
    main()
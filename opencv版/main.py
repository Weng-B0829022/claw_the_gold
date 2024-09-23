import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import math
import time
import pytesseract
from PIL import Image
import pyautogui

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from pynput import keyboard

# 全局变量来控制程序运行
running = True

def on_press(key):
    global running
    if key == keyboard.KeyCode.from_char('q'):
        print("检测到 'q' 键被按下，程序即将停止...")
        running = False
        return False  # 停止监听

def detect_and_click_button(window):
    global running
    
    # 启动键盘监听
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while running:
        # 获取窗口的位置
        left, top = window.left, window.top

        # 直接使用 pyautogui 获取指定区域的文字
        text = pyautogui.screenshot(region=(left, top, window.width, window.height))
        
        # 进行OCR，使用默认配置
        text = pytesseract.image_to_string(text, lang='chi_tra', config='--psm 6')
        
        # 将文字按行分割并去除空白行
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # 打印检测到的文字
        print("檢測到的文字:")
        for i, line in enumerate(lines, 1):
            print(f"{i}. {line}")
        
        # 寻找包含 "积分" 的文字
        target_word = "積"
        
        if target_word in text:
            # 找到 "积分"，点击指定坐标
            x = left + 200
            y = top + 510
            print(f"找到 '{target_word}'， ({x}, {y})...")
            pyautogui.click(x, y)
            
            # 进行40秒的连续点击，每0.5秒点击一次
            print("連續點擊45秒開始...")
            start_time = time.time()
            while time.time() - start_time < 45 and running:
                pyautogui.click(x, y)
                time.sleep(0.5)
            
            if running:
                print("連續點擊完成")
            else:
                print("連續點擊中斷")

            print("等待555秒")
            time.sleep(555)
        else:
            print(f"未找到包含 '{target_word}' 的文字")
        
        # 等待1秒后重复过程
        if running:
            time.sleep(1)

    print("程式已停止")


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


def click_window_center(window):
    left, top, width, height = window.left, window.top, window.width, window.height
    center_x = left + width // 2
    center_y = top + height * 3/4

    print(f"開始點擊視窗 '{window.title}' 的中心。按 'q' 鍵停止。")

    try:
        while True:
            # 模擬點擊窗口中心
            pyautogui.click(center_x, center_y)
            print(f"點擊位置 ({center_x}, {center_y})")

            # 檢查是否按下 'q' 鍵
            if keyboard.is_pressed('q'):
                print("檢測到 'q' 鍵被按下，停止點擊。")
                break

            # 等待一秒
            time.sleep(0.1)
    except Exception as e:
        print(f"發生錯誤: {e}")
    finally:
        print("點擊已停止。")

# 使用示例
# window = list_and_select_window()  # 假設你有這個函數來選擇窗口
# click_window_center(window)

def list_and_select_window():
    windows = [win for win in gw.getAllWindows() if win.title]
    print("當前打開的視窗:")
    for i, window in enumerate(windows, 1):
        print(f"{i}. {window.title}")
    
    while True:
        try:
            choice = int(input("請選擇視窗: "))
            if 1 <= choice <= len(windows):
                return windows[choice - 1]
            else:
                print("無效的選擇")
        except ValueError:
            print("請輸入數字")

def main():
    selected_window = list_and_select_window()
    print(f"您選了: {selected_window.title}")
    #detect_objects_in_window(selected_window)
    #click_window_center(selected_window)
    detect_and_click_button(selected_window)

if __name__ == "__main__":
    main()
import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import math

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
    selected_window = list_and_select_window()
    print(f"您选择了: {selected_window.title}")
    detect_objects_in_window(selected_window)

if __name__ == "__main__":
    main()
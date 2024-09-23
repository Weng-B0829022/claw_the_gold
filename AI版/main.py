import pygetwindow as gw
import numpy as np
import cv2
from ultralytics import YOLO
import os
import pyautogui
import time

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

def capture_window(window):
    left, top, width, height = window.left, window.top, window.width, window.height
    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return frame

def main():
    model_path = os.path.join(os.getcwd(), 'weights', 'final_model.pt')
    if not os.path.exists(model_path):
        print(f"找不到模型文件: {model_path}")
        return
    
    model = YOLO(model_path)

    selected_window = list_and_select_window()
    print(f"您選了: {selected_window.title}")

    print("開始持續檢測。按 'q' 鍵停止。")

    while True:
        image = capture_window(selected_window)
        results = model(image)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Window Recognition", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.01)  # 短暫暫停以減少CPU使用率

    cv2.destroyAllWindows()
    print("檢測已停止")

if __name__ == "__main__":
    main()
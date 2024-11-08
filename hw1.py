import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore


class Ui_Hw1(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi()
        self.image = None


    def setupUi(self):
        self.resize(800, 750)
        
        # Group Box 1 - Image Processing
        self.groupBox = QtWidgets.QGroupBox("1.Image processing", self)
        self.groupBox.setGeometry(140, 30, 225, 200)

        self.pushButton_3 = QtWidgets.QPushButton("1.1 Color separation", self.groupBox)
        self.pushButton_3.setGeometry(25, 40, 160, 25)
        self.pushButton_3.clicked.connect(self.click_color_separation)

        self.pushButton_4 = QtWidgets.QPushButton("1.2 Color transformation", self.groupBox)
        self.pushButton_4.setGeometry(25, 80, 160, 25)
        self.pushButton_4.clicked.connect(self.click_color_transformation)

        self.pushButton_5 = QtWidgets.QPushButton("1.3 Color extraction", self.groupBox)
        self.pushButton_5.setGeometry(25, 120, 160, 25)
        self.pushButton_5.clicked.connect(self.click_color_extraction)

        # Group Box 2 - Image Smoothing
        self.groupBox_2 = QtWidgets.QGroupBox("2.Image smoothing", self)
        self.groupBox_2.setGeometry(140, 250, 225, 200)

        self.pushButton_6 = QtWidgets.QPushButton("2.1 Gaussian blur", self.groupBox_2)
        self.pushButton_6.setGeometry(25, 40, 160, 25)
        self.pushButton_6.clicked.connect(self.click_gaussian)

        self.pushButton_7 = QtWidgets.QPushButton("2.2 Bilateral filter", self.groupBox_2)
        self.pushButton_7.setGeometry(25, 80, 160, 25)
        self.pushButton_7.clicked.connect(self.click_bilateral)

        self.pushButton_8 = QtWidgets.QPushButton("2.3 Median filter", self.groupBox_2)
        self.pushButton_8.setGeometry(25, 120, 160, 25)
        self.pushButton_8.clicked.connect(self.click_median)

        # Group Box 3 - Edge Detection
        self.groupBox_3 = QtWidgets.QGroupBox("3.Edge detection", self)
        self.groupBox_3.setGeometry(140, 480, 225, 235)

        self.pushButton_9 = QtWidgets.QPushButton("3.1 Sobel X", self.groupBox_3)
        self.pushButton_9.setGeometry(25, 25, 160, 25)
        self.pushButton_9.clicked.connect(self.click_smooth_x)

        self.pushButton_10 = QtWidgets.QPushButton("3.2 Sobel Y", self.groupBox_3)
        self.pushButton_10.setGeometry(25, 60, 160, 25)
        self.pushButton_10.clicked.connect(self.click_smooth_y)
        
        self.pushButton_11 = QtWidgets.QPushButton("3.3 Combination and threshold", self.groupBox_3)
        self.pushButton_11.setGeometry(20, 100, 160, 25)
        self.pushButton_11.clicked.connect(self.click_combiandthr)

        self.pushButton_12 = QtWidgets.QPushButton("3.4 Gradient angle", self.groupBox_3)
        self.pushButton_12.setGeometry(20, 140, 160, 25)
        self.pushButton_12.clicked.connect(self.gradient_angle)


        # Group Box 4 - Transforms
        self.groupBox_4 = QtWidgets.QGroupBox("4.Transforms", self)
        self.groupBox_4.setGeometry(400, 30, 350, 400)
        self.pushButton_13 = QtWidgets.QPushButton("4.Transforms", self.groupBox_4)
        self.pushButton_13.setGeometry(75, 300, 200, 30)
        self.pushButton_13.clicked.connect(self.click_transf_4)

        # Labels and LineEdits inside GroupBox 4
        labels = ["Rotation:", "Scaling:", "Tx:", "Ty:"]
        for i, text in enumerate(labels):
            label = QtWidgets.QLabel(text, self.groupBox_4)
            label.setGeometry(25, 45 + i*60, 56, 15)
        
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_4)
        self.lineEdit.setGeometry(100, 40, 141, 30)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox_4)
        self.lineEdit_2.setGeometry(100, 100, 141, 30)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.groupBox_4)
        self.lineEdit_3.setGeometry(100, 160, 141, 30)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.groupBox_4)
        self.lineEdit_4.setGeometry(100, 220, 141, 30)
        
        # Additional Labels for units
        self.label_5 = QtWidgets.QLabel("deg", self.groupBox_4)
        self.label_5.setGeometry(250, 40, 45, 18)
        self.label_6 = QtWidgets.QLabel("pixel", self.groupBox_4)
        self.label_6.setGeometry(250, 160, 45, 18)
        self.label_7 = QtWidgets.QLabel("pixel", self.groupBox_4)
        self.label_7.setGeometry(250, 220, 45, 18)

        # Load Image Buttons
        self.pushButton = QtWidgets.QPushButton("Load image1", self)
        self.pushButton.setGeometry(25, 250, 100, 45)
        self.pushButton.clicked.connect(self.click_load_1)
        self.pushButton_2 = QtWidgets.QPushButton("Load image2", self)
        self.pushButton_2.setGeometry(25, 425, 100, 45)
        self.pushButton_2.clicked.connect(self.click_load_2)

    #-------------------------------------------------------------------------------------------------------

    def click_load_1(self):
        options = QtWidgets.QFileDialog.Options()
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "選擇圖片檔案", "", "Images (*.png *.jpg *.bmp)", options=options)
        if filePath:
            self.image = cv2.imread(filePath)
            print("圖片已載入:", filePath)


    def click_load_2(self):
        options = QtWidgets.QFileDialog.Options()
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "選擇圖片檔案", "", "Images (*.png *.jpg *.bmp)", options=options)
        if filePath:
            self.image = cv2.imread(filePath)
            print("圖片已載入:", filePath)

    # color seperation
    def click_color_separation(self):
        if self.image is not None:
            b, g, r = cv2.split(self.image)
            k = np.zeros_like(b)
            b_image = cv2.merge([b, k, k]) 
            g_image = cv2.merge([k, g, k]) 
            r_image = cv2.merge([k, k, r])
            
            cv2.imshow("blue",b_image)
            cv2.imshow("green",g_image)
            cv2.imshow("red",r_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def click_color_transformation(self):
        if self.image is not None:
            b, g, r = cv2.split(self.image)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            avg_gray = (b/3 + g/3 + r/3).astype(np.uint8)
            cv2.imshow("Gray", gray)
            cv2.imshow("Avg_gray", avg_gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def click_color_extraction(self):
        if self.image is not None:
            hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            yellow_lower = np.array([19, 10, 0])
            yellow_upper = np.array([35, 255, 255])
            green_lower = np.array([36, 10, 0])
            green_upper = np.array([89, 255, 255])

            yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
            green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

            mask = cv2.bitwise_or(yellow_mask, green_mask)
            cv2.imshow("I1", mask)
            mask_inversed = cv2.bitwise_not(mask)
            extracted_image = cv2.bitwise_and(self.image, self.image, mask=mask_inversed)
            cv2.imshow("I2", extracted_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    
    def click_gaussian(self):
        if self.image is not None:
            cv2.namedWindow("Gaussian blur")
            def apply_gaussian_blur(m):
                m = max(0, m)
                kernal_size = 2 * m + 1 
                blurred_image = cv2.GaussianBlur(self.image, (kernal_size, kernal_size), sigmaX=0, sigmaY=0)
                cv2.imshow("Gaussian blur", blurred_image)

            cv2.createTrackbar("m value", "Gaussian blur", 0, 5, apply_gaussian_blur)
            apply_gaussian_blur(0)   
            cv2.waitKey(0)
            cv2.destroyAllWindows


    def click_bilateral(self):
        if self.image is not None:
            cv2.namedWindow("bilateral filter")
            def apply_bilateral(m):
                m = max(0, m)
                kernal_size = 2 * m + 1 
                bilateral_image = cv2.bilateralFilter(self.image, kernal_size, sigmaColor=90, sigmaSpace=90)
                cv2.imshow("bilateral filter", bilateral_image)

            cv2.createTrackbar("m value", "bilateral filter", 0, 5, apply_bilateral)
            apply_bilateral(0)   
            cv2.waitKey(0)
            cv2.destroyAllWindows
    

    def click_median(self):
        if self.image is not None:
            cv2.namedWindow("median blur")
            def apply_median(m):
                m = max(0, m)
                kernal_size = (2 * m + 1)*(2 * m + 1) 
                bilateral_image = cv2.medianBlur(self.image, kernal_size)
                cv2.imshow("median blur", bilateral_image)

            cv2.createTrackbar("m value", "median blur", 0, 5, apply_median)
            apply_median(0)   
            cv2.waitKey(0)
            cv2.destroyAllWindows

    
    def click_smooth_x(self):
        if self.image is not None:
            gray_3 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            gray_smoothed_x = cv2.GaussianBlur(gray_3, (3, 3), sigmaX=0, sigmaY=0)
            
            # 定義3x3的Sobel X運算元
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            # 初始化結果
            sobel_x_image = np.zeros_like(gray_smoothed_x, dtype=np.float32)

            # 使用卷積操作計算垂直邊緣
            for i in range(1, gray_smoothed_x.shape[0] - 1):
                for j in range(1, gray_smoothed_x.shape[1] - 1):
                    region = gray_smoothed_x[i - 1:i + 2, j - 1:j + 2]
                    gx = np.sum(sobel_x * region)
                    sobel_x_image[i, j] = abs(gx)
    
            # 將結果標準化到範圍0-255
            sobel_x_image = np.clip(sobel_x_image, 0, 255).astype(np.uint8)
            cv2.imshow("Sobel X", sobel_x_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    
    def click_smooth_y(self):
        if self.image is not None:
            gray_4 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # 高斯平滑處理
            gray_smoothed_y = cv2.GaussianBlur(gray_4, (3, 3), sigmaX=0, sigmaY=0)
    
            # Sobel Y 運算元
            sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
            # 初始化結果
            sobel_y_image = np.zeros_like(gray_smoothed_y, dtype=np.float32)
            # 手動卷積操作
            for i in range(1, gray_smoothed_y.shape[0] - 1):
                for j in range(1, gray_smoothed_y.shape[1] - 1):
                    region = gray_smoothed_y[i - 1:i + 2, j - 1:j + 2]
                    gy = np.sum(sobel_y * region)
                    sobel_y_image[i, j] = abs(gy)
    
            # 正規化範圍至 0-255
            sobel_y_image = np.clip(sobel_y_image, 0, 255).astype(np.uint8)
            cv2.imshow("Sobel Y", sobel_y_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            
    def click_combiandthr(self):
        if self.image is not None:
            gray_5 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            gray_smoothed_c = cv2.GaussianBlur(gray_5, (3, 3), sigmaX=0, sigmaY=0)

            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
            sobel_x_image = np.zeros_like(gray_smoothed_c, dtype=np.float32)
            sobel_y_image = np.zeros_like(gray_smoothed_c, dtype=np.float32)
            combined_image = np.zeros_like(gray_smoothed_c, dtype=np.float32)

            # 手動卷積操作，計算 Sobel X 和 Sobel Y
            for i in range(1, gray_smoothed_c.shape[0] - 1):
                for j in range(1, gray_smoothed_c.shape[1] - 1):
                    region = gray_smoothed_c[i - 1:i + 2, j - 1:j + 2]
                    gx = np.sum(sobel_x * region)
                    gy = np.sum(sobel_y * region)
                    sobel_x_image[i, j] = abs(gx)
                    sobel_y_image[i, j] = abs(gy)
                    combined_image[i, j] = np.sqrt(gx**2 + gy**2)
    
            # 正規化範圍至 0-255
            combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
    
            # 設定閾值並進行二值化處理
            threshold_128 = np.where(combined_image >= 128, 255, 0).astype(np.uint8)
            threshold_28 = np.where(combined_image >= 28, 255, 0).astype(np.uint8)
    
            # 顯示結果
            cv2.imshow("Threshold 128", threshold_128)
            cv2.imshow("Threshold 28", threshold_28)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def gradient_angle(self):
        if self.image is not None:
            gray_6 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            gray_smoothed_c = cv2.GaussianBlur(gray_6, (3, 3), sigmaX=0, sigmaY=0)

            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
            sobel_x_image = np.zeros_like(gray_smoothed_c, dtype=np.float32)
            sobel_y_image = np.zeros_like(gray_smoothed_c, dtype=np.float32)
            combined_image = np.zeros_like(gray_smoothed_c, dtype=np.float32)

            # 手動卷積操作，計算 Sobel X 和 Sobel Y
            for i in range(1, gray_smoothed_c.shape[0] - 1):
                for j in range(1, gray_smoothed_c.shape[1] - 1):
                    region = gray_smoothed_c[i - 1:i + 2, j - 1:j + 2]
                    gx = np.sum(sobel_x * region)
                    gy = np.sum(sobel_y * region)
                    sobel_x_image[i, j] = abs(gx)
                    sobel_y_image[i, j] = abs(gy)
                    combined_image[i, j] = np.sqrt(gx**2 + gy**2)
    
            # 正規化範圍至 0-255
            sobel_x_image = np.clip(sobel_x_image, 0, 255).astype(np.uint8)
            sobel_y_image = np.clip(sobel_y_image, 0, 255).astype(np.uint8)
            combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
            
            gradient_angle = np.arctan2(sobel_y_image, sobel_x_image) * (180 / np.pi)
            # 生成遮罩1：角度範圍170˚到190˚
    

            mask1 = np.where((gradient_angle >= (170 * np.pi / 180)) & (gradient_angle <= (190* np.pi / 180)), 255, 0).astype(np.uint8)
            # 生成遮罩2：角度範圍260˚到280˚
            mask2 = np.where((gradient_angle >= (260 * np.pi / 180)) & (gradient_angle <= (280* np.pi / 180)), 255, 0).astype(np.uint8)
            
            
            # 使用 cv2.bitwise_and 生成結果
            result1 = cv2.bitwise_and(combined_image, combined_image, mask=mask1)
            result2 = cv2.bitwise_and(combined_image, combined_image, mask=mask2)

            # 顯示結果
            cv2.imshow("Gradient Angle 170~190", result1)
            cv2.imshow("Gradient Angle 260~280", result2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def click_transf_4(self):
        if self.image is not None:

            
            # 從 line edit 獲取輸入值
            rotation_t = self.lineEdit.text()
            scaling_t = self.lineEdit_2.text()
            tx_t = self.lineEdit_3.text()
            ty_t = self.lineEdit_4.text()

            # 字串轉成數值
            rotation = int(rotation_t) if rotation_t else 0
            scaling =  float(scaling_t) if scaling_t else 1.0
            tx = int(tx_t) if tx_t else 0
            ty = int(ty_t) if ty_t else 0


            height, width, _ = self.image.shape

            # 計算中心點
            center_x = 240
            center_y = 200

            # 1. 建立旋轉矩陣（角度 = 30 度，按比例縮放 = 0.9）
            angle_rad = np.radians(30)
            M_rotation = np.array([
                [np.cos(angle_rad), np.sin(angle_rad), 0],
                [-np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])

            # 2. 建立縮放矩陣（比例 = 0.9）
            M_scale = np.array([
                [0.9, 0, 0],
                [0, 0.9, 0],
                [0, 0, 1]
            ])

            # 3. 建立平移矩陣（Tx = 535, Ty = 335）
            M_translation = np.array([
                [1, 0, 535],
                [0, 1, 335],
                [0, 0, 1]
            ])

            # 4. 合併仿射變換矩陣（M' = M_translation * M_scale * M_rotation）
            M_prime = M_translation @ M_scale @ M_rotation

            # 取前 2 行作為 cv2.warpAffine 所需的 2x3 矩陣
            M_prime_2x3 = M_prime[:2, :]

            # 應用仿射變換
            result = cv2.warpAffine(self.image, M_prime_2x3, (1920, 1080))

            # 顯示結果
            cv2.imshow("Transformed Image", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet("QWidget { font-size: 6pt; }")
    window = Ui_Hw1()
    window.show()
    sys.exit(app.exec_())

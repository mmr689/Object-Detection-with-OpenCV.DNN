import os
import cv2


def main(img_path, model):
    architecture, weights, classes = model
    # ------- LOAD THE MODEL -------
    net = cv2.dnn.readNetFromCaffe(architecture, weights)

    # ------- READ THE IMAGE AND PREPROCESSING -------
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    img_resized = cv2.resize(img, (300, 300))
    # Create a blob
    blob = cv2.dnn.blobFromImage(img_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))
    
    # ------- DETECTIONS AND PREDICTIONS ----------
    net.setInput(blob)
    detections = net.forward()
    # Work with all object detections
    for detection in detections[0][0]:
        # Only show detection with 45% confidence
        if detection[2] > 0.45:
            label = classes[detection[1]]
            box = detection[3:7] * [w, h, w, h]
            x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(img, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 5), 1, 1.2, (0, 255, 255), 2)
            cv2.putText(img, label, (x_start, y_start - 25), 1, 1.2, (255, 0, 0), 2)

    # ------- SHOW RESULTS -------
    cv2.imshow("Image", img)

    # ------- IF SOME KEY PRESSED CLOSE IMAGE -------
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # ------- WORK PATHS -------
    current_dir = os.getcwd()
    # DNN
    architecture = os.path.join(current_dir, 'myFiles', 'models', 'MobileNetSSD_deploy.prototxt')
    weights = os.path.join(current_dir, 'myFiles', 'models', 'MobileNetSSD_deploy.caffemodel')
    # Images
    img_list = ['test1.jpg','test2.jpg','test3.jpg', 'test4.jpg']
    path_in_list = [os.path.join(current_dir, 'myFiles', 'imgs', file_name) for file_name in img_list]


    # ------- Class labels -------
    classes = {0:"background", 1:"aeroplane", 2:"bicycle", 3:"bird", 4:"boat",
            5:"bottle", 6:"bus", 7:"car", 8:"cat", 9:"chair", 10:"cow",
            11:"diningtable", 12:"dog", 13:"horse", 14:"motorbike", 15:"person",
            16:"pottedplant", 17:"sheep", 18:"sofa", 19:"train", 20:"tvmonitor"}
    # -------  MAIN -------
    for path in path_in_list:
        main(img_path=path, model=(architecture, weights, classes))

    print(' *** END PROGRAM ***\n')
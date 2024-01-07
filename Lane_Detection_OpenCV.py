import cv2
import numpy as np
import time

cap = cv2.VideoCapture('img\Car_Driving.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Video ended. Exiting ...")
        break
    
    start = time.perf_counter()
    
    # Open image and apply gray- and Gaussian blur filter to remove noise
    gray = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    # Applying Canny filter to create an image that shows all the edges
    edges = cv2.Canny(blur, 150, 200)
    
    # Create a ROI (Regen Of Interest) of the picture so only the lanes are displayed
    # This is done by masking a trapezium shaped polygon over the ROI and filter the rest away

    # create an array of the same size as of the input image 
    mask = np.zeros_like(edges)

    # creating a trapezium polygon to focus only on the road in the picture
    rows, cols = edges.shape[:2]
    bottom_left  = [cols * 0.1, rows * 1]
    top_left     = [cols * 0.43, rows * 0.72]
    bottom_right = [cols * 0.9, rows * 1]
    top_right    = [cols * 0.55, rows * 0.72]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    # filling the polygon with white color and generating the final mask
    cv2.fillPoly(mask, vertices, 255)
    # performing Bitwise AND on the input image and mask to get only the edges on the road
    masked_image = cv2.bitwise_and(edges, mask)

    # Drawing lines via Hough transform coördinates on the masked image
    # Copy those lines and paste them on the original image
    original_img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    Arr_slope = []
    Arr_slope_right = []
    Arr_slope_left = []

    lines = cv2.HoughLinesP(masked_image, 1, np.pi / 180, 20, None, 20, 500)
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            #slope =        (y2 - y1)/(x2 - x1)
            if (l[2]-l[0]) != 0:
                slope = round((l[3]-l[1])/(l[2]-l[0]),1)
                if slope not in Arr_slope:
                    if (slope > -1 and slope < -0.5):
                        # Negative slope = Left road lines
                        Arr_slope_left.append((l[0],l[1],l[2],l[3]))
                    elif (slope < 1 and slope > 0.5):
                        # Positive slope = Right road lines
                        Arr_slope_right.append((l[0],l[1],l[2],l[3]))

                Arr_slope.append(slope)

        if len(Arr_slope_left) > 0:
            # Take average of left lines
            Avg_line_left = np.average(Arr_slope_left, axis=0)
            Avg_line_left = np.split(Avg_line_left,4)

            # Extend average line to the bottom of the image
            vector = (int(Avg_line_left[2][0])-int(Avg_line_left[0][0]) , int(Avg_line_left[3][0])-int(Avg_line_left[1][0]))
            Meanvector = np.sqrt(np.power(vector[0],2)+np.power(vector[1],2))
            NormalisedVectorX = int(vector[0]) / float(Meanvector)
            NormalisedVectorY = int(vector[1]) / float(Meanvector)

            # Draw the extended left line
            cv2.line(original_img, (int(Avg_line_left[0][0])-int(350*NormalisedVectorX), int(Avg_line_left[1][0])-int(350*NormalisedVectorY)), (int(Avg_line_left[2][0])+int(5*NormalisedVectorX), int(Avg_line_left[3][0])+int(5*NormalisedVectorY)), (0, 0, 255), 12)
            #cv2.line(original_img, (int(Avg_line_left[0][0]), int(Avg_line_left[1][0])), (int(Avg_line_left[2][0]), int(Avg_line_left[3][0])), (0, 0, 255), 12)

        if len(Arr_slope_right) > 0:
            # Take average of right lines
            Avg_line_right = np.average(Arr_slope_right, axis=0)
            Avg_line_right = np.split(Avg_line_right,4)

            # Extend average line to the bottom of the image
            vector = (int(Avg_line_right[2][0])-int(Avg_line_right[0][0]) , int(Avg_line_right[3][0])-int(Avg_line_right[1][0]))
            Meanvector = np.sqrt(np.power(vector[0],2)+np.power(vector[1],2))
            NormalisedVectorX = int(vector[0]) / float(Meanvector)
            NormalisedVectorY = int(vector[1]) / float(Meanvector)

            # Draw the extended right line
            cv2.line(original_img, (int(Avg_line_right[0][0])-int(5*NormalisedVectorX), int(Avg_line_right[1][0])-int(5*NormalisedVectorY)), (int(Avg_line_right[2][0])+int(350*NormalisedVectorX), int(Avg_line_right[3][0])+int(350*NormalisedVectorY)), (0, 0, 255), 12)
            #cv2.line(original_img, (int(Avg_line_right[0][0]), int(Avg_line_right[1][0])), (int(Avg_line_right[2][0]), int(Avg_line_right[3][0])), (0, 0, 255), 12)

    #cv2.polylines(original_img, [vertices], True, (0,255,0), thickness=3)
    
    end = time.perf_counter()
    
    cv2.imshow("Result", original_img)
    
    print("Elapsed = {}ms".format((end - start)*1000))
    
    if cv2.waitKey(25) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

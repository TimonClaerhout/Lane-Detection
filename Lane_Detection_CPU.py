import numpy as np
import cv2 as cv

import time

cap = cv.VideoCapture('img\Car_Driving.mp4')

'''
cv.namedWindow("Result", cv.WINDOW_NORMAL)
cv.setWindowProperty("Result", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
'''

def image_filter_gaussian_cpu(input_image, output_image):
    width,height = input_image.shape[:2]

    for x in range(width):
        for y in range(height):
            if x < input_image.shape[0] - 2 and y < input_image.shape[1] - 2:
                output_image[x + 1, y + 1] = (4 * input_image[x + 1,y + 1] + input_image[x,y] + 2 * input_image[x + 1,y] + input_image[x + 2,y] + 2 * input_image[x,y + 1] + 2 * input_image[x + 2,y + 1] + input_image[x,y + 2] + 2 * input_image[x + 1,y + 2] + input_image[x + 2,y + 2])/16

def convolve_cpu(input_image, mask, output_image):
    width,height = input_image.shape[:2]
    m_width, m_height = mask.shape[:2]
    
        
    delta_rows = m_width // 2 
    delta_cols = m_height // 2
        
    for x in range(width):
        for y in range(height):

            s = 0
            for k in range(m_width):
                for l in range(m_height):
                    x_k = x - k + delta_rows
                    y_l = y - l + delta_cols
                    # (-4-) Check if (i_k, j_k) coordinates are inside the image: 
                    if (x_k >= 0) and (x_k < input_image.shape[0]) and (y_l >= 0) and (y_l < input_image.shape[1]):  
                        s += mask[k, l] * input_image[x_k, y_l]
            output_image[x, y] = s

def image_filter_sobel_cpu(conv_img_x, conv_img_y, output_image, angle):
    width,height = conv_img_x.shape[:2]

    for x in range(width):
        for y in range(height):  
            output_image[x,y] = (conv_img_x[x,y]**2 + conv_img_y[x,y]**2)**0.5
            output_image[x,y] *= 255.0 / 360
            
            # angle is used for the next filter.
            # angle = theta * 180/pi
            angle[x,y] = (np.arctan2(conv_img_y[x,y],conv_img_x[x,y]))*180.0/np.pi
            if angle[x,y] < 0:
                angle[x,y] += 180

       
def image_filter_non_max_cpu(input_image, angle, output_image): 
    q = 255
    r = 255
    width,height = input_image.shape[:2]

    for x in range(width-1):
        for y in range(height-1):  

            if (0 <= angle[x,y] < 22.5) or (157.5 <= angle[x,y] <= 180):
                r = input_image[x,y-1]
                q = input_image[x,y+1]
        
            elif (22.5 <= angle[x,y] < 67.5):
                r = input_image[x-1, y+1]
                q = input_image[x+1, y-1]
        
            elif (67.5 <= angle[x,y] < 112.5):
                r = input_image[x-1, y]
                q = input_image[x+1, y]
        
            elif (112.5 <= angle[x,y] < 157.5):
                r = input_image[x+1, y+1]
                q = input_image[x-1, y-1]
        
            if (input_image[x,y] >= q) and (input_image[x,y] >= r):
                output_image[x,y] = input_image[x,y]
            else:
                output_image[x,y] = 0
        
        
def image_black_white_filter_cpu(input_image, output_image):
    width,height = input_image.shape[:2]

    highThreshold = 255
    lowThreshold = 127
    for x in range(width):
        for y in range(height):          
            if (input_image[x, y] > lowThreshold):
                output_image[x, y] = highThreshold
            else:
                output_image[x, y] = 0
      
while True:

    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    
    
    image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Allocate memory for the output image
    black_white_image = np.zeros_like(image)
    gauss_image = np.zeros_like(image)
    sobel_image = np.zeros_like(image)
    canny_image = np.zeros_like(image)
    
    conv_img_x = np.zeros_like(image)
    conv_img_y = np.zeros_like(image)
    angle = np.zeros_like(image)
    
    # Create mask for convolve
    maskX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    maskY = np.array([[-1,-2, -1], [0, 0, 0], [1, 2, 1]])
    
     
    start = time.perf_counter()
    
    image_black_white_filter_cpu(image, black_white_image)  
    
    # Apply filter
    image_filter_gaussian_cpu(black_white_image, gauss_image)
    

    convolve_cpu(gauss_image, maskX, conv_img_x)
    convolve_cpu(gauss_image, maskY, conv_img_y)

    image_filter_sobel_cpu(conv_img_x, conv_img_y, sobel_image, angle)

    image_filter_non_max_cpu(sobel_image,angle,canny_image)
    
    #cv.imshow("Result", sobel_image)
    
    
    # Create a ROI (Regen Of Interest) of the picture so only the lanes are displayed
    # This is done by masking a trapezium shaped polygon over the ROI and filter the rest away

    # create an array of the same size as of the input image 
    ROI_mask = np.zeros_like(canny_image)

    # creating a triangle shaped polygon to focus only on the road in the picture
    rows, cols = canny_image.shape[:2]
    bottom_left  = [cols * 0.1, rows * 1]
    top_left     = [cols * 0.43, rows * 0.72]
    bottom_right = [cols * 0.9, rows * 1]
    top_right    = [cols * 0.55, rows * 0.72]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    # filling the polygon with white color and generating the final mask
    cv.fillPoly(ROI_mask, vertices, 255)
    # performing Bitwise AND on the input image and mask to get only the edges on the road
    masked_canny_image = cv.bitwise_and(canny_image, ROI_mask)

    # Drawing lines via Hough transform coördinates on the masked image
    # Copy those lines and paste them on the original image
    original_img = cv.cvtColor(frame, cv.IMREAD_COLOR)
    Arr_slope = []
    Arr_slope_right = []
    Arr_slope_left = []
    
    lines = cv.HoughLinesP(masked_canny_image, 1, np.pi / 180, 20, None, 20, 500)
    
    ''' UNCOMMENT TO DISPLAY ALL HOUGH LINES
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv.line(original_img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    '''
    
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
            Meanvector = (vector[0]**2+vector[1]**2)**0.5
            NormalisedVectorX = int(vector[0]) / float(Meanvector)
            NormalisedVectorY = int(vector[1]) / float(Meanvector)

            # Draw the extended left line
            cv.line(original_img, (int(Avg_line_left[0][0])-int(250*NormalisedVectorX), int(Avg_line_left[1][0])-int(250*NormalisedVectorY)), (int(Avg_line_left[2][0])+int(NormalisedVectorX), int(Avg_line_left[3][0])+int(NormalisedVectorY)), (0, 0, 255), 12)
            
            # Draw the left line
            #cv.line(original_img, (int(Avg_line_left[0][0]), int(Avg_line_left[1][0])), (int(Avg_line_left[2][0]), int(Avg_line_left[3][0])), (0, 0, 255), 12)

        if len(Arr_slope_right) > 0:
            # Take average of right lines
            Avg_line_right = np.average(Arr_slope_right, axis=0)
            Avg_line_right = np.split(Avg_line_right,4)

            # Extend average line to the bottom of the image
            vector = (int(Avg_line_right[2][0])-int(Avg_line_right[0][0]) , int(Avg_line_right[3][0])-int(Avg_line_right[1][0]))
            Meanvector = (vector[0]**2+vector[1]**2)**0.5
            NormalisedVectorX = int(vector[0]) / float(Meanvector)
            NormalisedVectorY = int(vector[1]) / float(Meanvector)

            # Draw the extended right line
            cv.line(original_img, (int(Avg_line_right[0][0])-int(NormalisedVectorX), int(Avg_line_right[1][0])-int(NormalisedVectorY)), (int(Avg_line_right[2][0])+int(250*NormalisedVectorX), int(Avg_line_right[3][0])+int(250*NormalisedVectorY)), (0, 0, 255), 12)

            # Draw the right line
            #cv.line(original_img, (int(Avg_line_right[0][0]), int(Avg_line_right[1][0])), (int(Avg_line_right[2][0]), int(Avg_line_right[3][0])), (0, 0, 255), 12)

    # Draw the ROI zone
    #cv.polylines(original_img, [vertices], True, (0,255,0), thickness=3)

    
    end = time.perf_counter()
    print("Elapsed = {}ms".format((end - start)*1000))
    
    cv.imshow("Result", original_img)
    
    
    if cv.waitKey(25) == ord('q'):
        break

# When everything done, release the capture
cap.release()
#cam2.release()
cv.destroyAllWindows()

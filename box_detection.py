from __future__ import division
import cv2
import numpy as np

def remove_negative(value):
    if(value < 0):
        return 0
    else:
        return value

def get_crop_vals(
        x1,
        x2,
        x3,
        x4,
        y1,
        y2,
        y3,
        y4,
        width_resize_scale,
        height_resize_scale):
        
    result_x1 = (x1 * width_resize_scale)
    result_x2 = result_x1 + (x2 * width_resize_scale)
    result_x3 = result_x1 + (x3 * width_resize_scale)
    result_x4 = result_x3 + (x4 * width_resize_scale)
    result_y1 = (y1 * height_resize_scale)
    result_y2 = result_y1 + (y2 * height_resize_scale)
    result_y3 = result_y2 + (y3 * height_resize_scale)
    result_y4 = result_y1 + (y4 * height_resize_scale)
    result = [
        remove_negative(
            int(result_x1)), remove_negative(
            int(result_y1)), remove_negative(
                int(result_x3)), remove_negative(
                    int(result_y3))]
    return result


def remove_text(img):
    upper_red = np.array([0,0,255])
    lower_red = np.array([0,0,0])
    mask1 = cv2.inRange(img,lower_red,upper_red )
    fin = cv2.bitwise_and(img, img, mask = mask1)
    cv2.imwrite("mask1.jpg",mask1)
    cv2.imwrite("fin.jpg",fin)
    fin_grey = cv2.cvtColor(fin, cv2.COLOR_BGR2GRAY)
    return fin_grey

def remove_boxes(img):
    upper_red = np.array([250,250,255])
    lower_red = np.array([100,100,100])
    mask1 = cv2.inRange(img,upper_red,lower_red )
    mask1_inv = cv2.bitwise_not(mask1)
    fin2 = cv2.bitwise_and(img, img, mask = mask1_inv)
    cv2.imwrite("mask1_inv.jpg",mask1_inv)
    cv2.imwrite("fin2.jpg",fin2)
    return fin2

def get_offset_val(x1,x3,y1,y3,height_scale,width_scale):
    off_set_dict = {

        "x1_offset" : int(round(x1/2)),
        "x2_offset" : int(round(0)),
        "x3_offset" : int(round((x3 - x1)/2)),
        "x4_offset" : int(round(0)),
        "y1_offset" : int(round(y1/2)),
        "y2_offset" : int(round((y3 - y1)/2)),
        "y3_offset" : int(round(0)),
        "y4_offset" : int(round(0))

    }
    return off_set_dict
    

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    
    return (cnts, boundingBoxes)

def box_extraction(img_for_box_extraction_path, cropped_dir_path):

    doc_height =  672
    doc_width = 454
    height_scale = 1 
    width_scale = 1 

    img_org = cv2.imread(img_for_box_extraction_path) 
    # img_org = cv2.resize(img_org, None, fx=0.40, fy=0.40) 
    img_cpy = img_org.copy()
    img_template = img_org.copy()
    img_height, img_width, img_dims =  img_org.shape
    height_scale = doc_height/img_height
    width_scale = doc_width/img_width

    print "image_height : {}, image_width : {}".format(img_height,img_width)

    print "height_scale : {},width_scale : {}, height : {}, width : {}".format(height_scale,width_scale,doc_height,doc_width)
    img = remove_text(img_org)

    (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)  
    img_bin = 255-img_bin  
    cv2.imwrite("Image_bin.jpg",img_bin)
   
    
    kernel_length = np.array(img).shape[1]//40
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=1)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=1)
    cv2.imwrite("verticle_lines.jpg",verticle_lines_img)

    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=1)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=1)
    cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)

    alpha = 0.5
    beta = 1.0 - alpha
    
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=1)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("img_final_bin.jpg",img_final_bin)
    
    im2, contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    idx = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 200 and h < 100:
            idx += 1
            new_img = img_org[y:y+h, x:x+w]
            cv2.rectangle(img_cpy, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.rectangle(img_template, (x-3, y-3), (x+w+2, y+h+2), (255, 255, 255), -1)
            cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)
            print(get_offset_val(x,x+w,y,y+h,height_scale,width_scale))
    
    x1 = 340
    x2 = 0
    x3 = 240
    x4 = 0   
    y1 = 735
    y2 = 40
    y3 = 0
    y4 = 0

    result = get_crop_vals(x1,x2,x3,x4,y1,y2,y3,y4,1/width_scale,1/height_scale)
    print result
    cv2.rectangle(img_cpy, (result[0], result[1]), (result[2], result[3]), (255, 0, 0), 3)
    cv2.imwrite("img_cpy" + '.png', img_cpy)
    cv2.imwrite("img_template" + '.png', img_template)

    
    
box_extraction("dec_form_squared.jpg", "./Cropped/")
from __future__ import division
import cv2
import numpy as np




def remove_text(img):
    upper_red = np.array([0,0,255])
    lower_red = np.array([0,0,0])
    mask1 = cv2.inRange(img,lower_red,upper_red )
    fin = cv2.bitwise_and(img, img, mask = mask1)
    cv2.imwrite("mask1.jpg",mask1)
    cv2.imwrite("fin.jpg",fin)
    fin_grey = cv2.cvtColor(fin, cv2.COLOR_BGR2GRAY)
    return fin_grey

def get_offset_val(x1,x3,y1,y3,height_scale,width_scale):
    off_set_dict = {

        "x1_offset" : int(round(x1 / width_scale)),
        "x2_offset" : int(round(0 / width_scale)),
        "x3_offset" : int(round((x3 - x1) / width_scale)),
        "x4_offset" : int(round(0 / width_scale)),
        "y1_offset" : int(round(y1 / height_scale)),
        "y2_offset" : int(round((y3 - y1) / height_scale)),
        "y3_offset" : int(round(0 / height_scale)),
        "y4_offset" : int(round(0 / height_scale))

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
    img_cpy = img_org.copy()
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
    
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    cv2.imwrite("verticle_lines.jpg",verticle_lines_img)

    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)

    alpha = 0.5
    beta = 1.0 - alpha
    
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("img_final_bin.jpg",img_final_bin)
    
    im2, contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    idx = 0
    for c in contours:
        
        x, y, w, h = cv2.boundingRect(c)
        idx += 1
        new_img = img_org[y:y+h, x:x+w]
        cv2.rectangle(img_cpy, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)
        print(get_offset_val(x,x+w,y,y+h,height_scale,width_scale))
    cv2.imwrite("img_cpy" + '.png', img_cpy)
    
    
box_extraction("red_square_dec_form2.jpeg", "./Cropped/")
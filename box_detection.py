from __future__ import division
import cv2
import numpy as np
import pytesseract

TESSDATA_PATH = "../CariPay/ocr/src/tessdata/"

def remove_negative(value):
    if(value < 0):
        return 0
    else:
        return value

def clean_box_with_text(img):
    img_inv = cv2.bitwise_not(img)
    th2 = cv2.adaptiveThreshold(img_inv,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    cv2.imwrite("th2.jpg",th2)
    cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR)
    return th2  

def ocr_img(img):
    return tesseract_process_img(img, 7, 300)

def get_tabs_as_spaces(tabs_sapces):
    spaces_string = ""
    for i in range(tabs_sapces):
        spaces_string = spaces_string + "  "
    return spaces_string


def generate_yaml_file(name,fields_dict,info_dict):
    yaml_file = open("{}.yaml".format(name),"w")
    yaml_file.write("company: {}".format(info_dict["company"]))
    yaml_file.write("\n")
    yaml_file.write("scale:")
    yaml_file.write("\n")
    yaml_file.write("{}height: {}".format(get_tabs_as_spaces(1),info_dict["height"]))
    yaml_file.write("\n")
    yaml_file.write("{}width: {}".format(get_tabs_as_spaces(1),info_dict["width"]))
    yaml_file.write("\n")
    yaml_file.write("fields:")
    yaml_file.write("\n")

    for key in fields_dict:
        yaml_file.write("{}{}:".format(get_tabs_as_spaces(1),key))
        yaml_file.write("\n")
        yaml_file.write("{}crop_vals:".format(get_tabs_as_spaces(2),key))
        yaml_file.write("\n")
        yaml_file.write("{}x1: {}".format(get_tabs_as_spaces(3),fields_dict[key]["x1_offset"]))
        yaml_file.write("\n")
        yaml_file.write("{}x2: {}".format(get_tabs_as_spaces(3),fields_dict[key]["x2_offset"]))
        yaml_file.write("\n")
        yaml_file.write("{}x3: {}".format(get_tabs_as_spaces(3),fields_dict[key]["x3_offset"]))
        yaml_file.write("\n")
        yaml_file.write("{}x4: {}".format(get_tabs_as_spaces(3),fields_dict[key]["x4_offset"]))
        yaml_file.write("\n")
        yaml_file.write("{}y1: {}".format(get_tabs_as_spaces(3),fields_dict[key]["y1_offset"]))
        yaml_file.write("\n")
        yaml_file.write("{}y2: {}".format(get_tabs_as_spaces(3),fields_dict[key]["y2_offset"]))
        yaml_file.write("\n")
        yaml_file.write("{}y3: {}".format(get_tabs_as_spaces(3),fields_dict[key]["y3_offset"]))
        yaml_file.write("\n")
        yaml_file.write("{}y4: {}".format(get_tabs_as_spaces(3),fields_dict[key]["y4_offset"]))
        yaml_file.write("\n")
        yaml_file.write("{}clean_vals:".format(get_tabs_as_spaces(2)))
        yaml_file.write("\n")
        yaml_file.write("{}intensity: 0".format(get_tabs_as_spaces(3)))
        yaml_file.write("\n")
        yaml_file.write("{}type: numbers".format(get_tabs_as_spaces(3)))
        yaml_file.write("\n")
        yaml_file.write("{}tesseract_vals:".format(get_tabs_as_spaces(2)))
        yaml_file.write("\n")
        yaml_file.write("{}psm: 7".format(get_tabs_as_spaces(3)))
        yaml_file.write("\n")
        yaml_file.write("{}dpi: 300".format(get_tabs_as_spaces(3)))
        yaml_file.write("\n")
    yaml_file.close()
    return "config file sucessfully generated"

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

def tesseract_process_img(img, psm, dpi):
    config_str = '--psm {0} --dpi {1} --tessdata-dir "{2}"'.format(
        psm, dpi, TESSDATA_PATH)
    return pytesseract.image_to_string(img, config=config_str)

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

def get_offset_val(x1,x3,y1,y3,height_scale,width_scale,title):
    off_set_dict = {
        "title" : title,
        "x1_offset" : int(round(x1)),
        "x2_offset" : int(round(0)),
        "x3_offset" : int(round((x3 - x1))),
        "x4_offset" : int(round(0)),
        "y1_offset" : int(round(y1)),
        "y2_offset" : int(round((y3 - y1))),
        "y3_offset" : int(round(0)),
        "y4_offset" : int(round(0))

    }
    return off_set_dict
def generate_info_dict(company,height,width):
    return {
    "company" : company,
    "height" : height,
    "width" : width
    }    

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
    company = "GUARDIAN"
    yaml_file_name = "guardian_declaration_form"
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

    info_dict = generate_info_dict(company,img_height,img_width)

    print "image_height : {}, image_width : {}".format(img_height,img_width)

    print "height_scale : {},width_scale : {}, height : {}, width : {}".format(height_scale,width_scale,doc_height,doc_width)
    img = remove_text(img_org)

    image_with_boxes_and_text = clean_box_with_text(img)

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
    fields_dict = {}
    idx = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 150 and h < 100:
            idx += 1
            x = x +3
            y = y +3 
            w = w -6 
            h = h -6 
            new_img = img_org[y:y+h, x:x+w]
            cropped_text = image_with_boxes_and_text[y:y+h, x:x+w]
            field_name = ocr_img(cropped_text)
            cv2.rectangle(img_cpy, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.rectangle(img_template, (x-5, y-5), (x+w+5, y+h+5), (255, 255, 255), -1)
            cv2.imwrite(cropped_dir_path+str(idx) + '{}.png'.format(field_name), new_img)
            cv2.imwrite(cropped_dir_path+str(idx) + '_text_{}.png'.format(field_name), cropped_text)
            if (field_name != "ignore"):
                fields_dict[field_name] = get_offset_val(x,x+w,y,y+h,height_scale,width_scale,field_name)
    x1 = 340
    x2 = 0
    x3 = 240
    x4 = 0   
    y1 = 735
    y2 = 40
    y3 = 0
    y4 = 0
    print(fields_dict)
    result = get_crop_vals(x1,x2,x3,x4,y1,y2,y3,y4,width_scale,height_scale)
    print result
    cv2.rectangle(img_cpy, (result[0], result[1]), (result[2], result[3]), (255, 0, 0), 3)
    cv2.imwrite("img_cpy" + '.png', img_cpy)
    cv2.imwrite("img_template" + '.png', img_template)
    print(generate_yaml_file(yaml_file_name,fields_dict,info_dict))
    
box_extraction("small_dec_template.jpg", "./Cropped/")
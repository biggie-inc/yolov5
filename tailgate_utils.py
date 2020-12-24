from numpy.core.numeric import full
from scipy.stats import mode
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_mode(color_image):
    grayscale = cv2.cvtColor(color_image.copy(), cv2.COLOR_BGR2GRAY)
    mode_of_image, count = mode(grayscale, axis=None)
    return int(mode_of_image[0])


def adaptive_histogram(image):
    grayscale = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
    cl1 = clahe.apply(grayscale)
    return cl1


def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def auto_canny(image, sigma=0.66):
    #compute median of single channel pixel intentsities
    med = np.median(image)

    #apply automatic canny edge detection using median and sigma
    lower = int(max(0, (1.0-sigma)*med))
    upper = int(min(255, (1.0+sigma)*med))
    edged = cv2.Canny(image, lower, upper, apertureSize=3)

    return edged


def layered_edge_detection(bilat):
    edges1 = auto_canny(bilat.copy())

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) 
    dilated = cv2.dilate(edges1.copy(), kernel) #dilating edges to connect segments

    edges2 = auto_canny(dilated.copy())

    return edges2


def get_contours(edged_image):
    cnts, _ = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(cnts, key=cv2.contourArea, reverse=True)

    return sorted_contours


def contours_from_edges_on_contours(image, sorted_contours):
    drawn_ctrs = cv2.drawContours(image.copy(), sorted_contours, -1, (0, 255, 0), 2) 

    edges3 = auto_canny(drawn_ctrs.copy())

    cntrs2, _ = cv2.findContours(edges3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    sorted_contours2 = sorted(cntrs2, key=cv2.contourArea)

    return sorted_contours2


def get_array_of_corners(image):    
    try:
        grayscale = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    except:
        grayscale = image.copy()

    # dst = cv2.cornerHarris(grayscale,2,5,0.04) # another corner alg that didn't work as well
    corners = cv2.goodFeaturesToTrack(grayscale,30,0.0001,20) #returns as float
    corners = np.int0(corners) # turns all floats to int

    #inspect_all_corners = image.copy() #For visualization

    for i in corners:
        x,y = i.ravel() #isn't ravel the coolest?
        #cv2.circle(inspect_all_corners,(x,y),3,255,-1) # places circles on all found corners
        #plt.imshow(inspect_all_corners)

    line_1 = int(image.shape[0] * 0.1)
    line_2 = int(image.shape[0] - line_1)
    line_3 = int(image.shape[1] - line_1)

    corners_array = []
    
    #final_corners = image.copy() # For visualization

    for i in corners:
        x,y = i.ravel()
        if x < line_1 and y < line_1:
            #cv2.circle(final_corners,(x,y),3,255,-1)
            corners_array.append([[x,y]])    
        elif x < line_1 and y > line_2:
            #cv2.circle(final_corners,(x,y),3,255,-1)
            corners_array.append([[x,y]])
        elif x > line_3 and y > line_2:
            #cv2.circle(final_corners,(x,y),3,255,-1)
            corners_array.append([[x,y]])
        elif x > line_3 and y < line_1:
            #cv2.circle(final_corners,(x,y),3,255,-1)
            corners_array.append([[x,y]])
    return np.array(corners_array)


def add_border(truck_image):
    bordered_image = cv2.copyMakeBorder(truck_image, 2,2,2,2, cv2.BORDER_CONSTANT, value=(0,255,0))
    return bordered_image

def remove_border(bordered_image):
    cropped = bordered_image[2:bordered_image.shape[0]-2, 2:bordered_image.shape[1]-2] #necessary to crop border
    return cropped

def border_process(bilat_image):
    border = add_border(bilat_image)
    edges = layered_edge_detection(border)
    cropped = remove_border(edges.copy())

    return cropped


def transparent_tailgate_mask(orig_image, hull):
    BGRA = cv2.cvtColor(orig_image.copy(), cv2.COLOR_BGR2BGRA)
    masked = cv2.drawContours(BGRA.copy(), [hull], -1, (0,0,0,0), -1)

    masked_image = cv2.bitwise_and(BGRA, masked)
    return masked_image


def transparent_handle_mask(orig_image, hull):
    BGRA = cv2.cvtColor(orig_image.copy(), cv2.COLOR_BGR2BGRA)
    mask = np.zeros(BGRA.shape, BGRA.dtype)

    cv2.fillPoly(mask, [hull], (255,)*BGRA.shape[2], )

    masked_image = cv2.bitwise_and(BGRA, mask)
    return masked_image


def get_tailgate_dims(image):
    #which x,y pairs aren't transparent. Outputs two arrays - [0]:y's, [1]:x's
    tg_coords = np.where(np.all((image == [0,0,0,0]),axis=-1)) 
    if len(tg_coords[0]) > 1 and len(tg_coords[1]) > 1:
        tg_ymin, tg_ymax = min(tg_coords[0]), max(tg_coords[0])
        tg_xmin, tg_xmax = min(tg_coords[1]), max(tg_coords[1])

        pixel_width = tg_xmax - tg_xmin
        pixel_height = tg_ymax - tg_ymin
    else:
        pixel_width = ''
        pixel_height = ''


    return pixel_width, pixel_height


def get_handle_dims(image):
    #which x,y pairs aren't transparent. Outputs two arrays - [0]:y's, [1]:x's
    handle_coords = np.where(np.all((image != [0,0,0,0]),axis=-1))
    if len(handle_coords[0]) > 1 and len(handle_coords[1]) > 1:
        handle_ymin, handle_ymax = min(handle_coords[0]), max(handle_coords[0])
        handle_xmin, handle_xmax = min(handle_coords[1]), max(handle_coords[1])
        pixel_width = handle_xmax - handle_xmin
        pixel_height = handle_ymax - handle_ymin
    else:
        pixel_width = ''
        pixel_height = ''
    return pixel_width, pixel_height


def tailgate_detect_and_mask(image):
    image_area = image.shape[0] * image.shape[1]
    mode_of_image = get_mode(image.copy())

    full_process = [f'mode of image: {mode_of_image}']

    if mode_of_image >= 125: # The mode of the image will determine if the image needs to be darkened or lightened
        first_range_tens = np.arange(0,-111, -10)
        full_process.append('darken process')
    else:
        first_range_tens = np.arange(0,111, 10)
        full_process.append('lighten process')

    second_range_tens = np.arange(0,111, 10)
   
    # Multiple sections of for loops used as edge detection should be primary,
    # Corner detection secondary
    # And border creation to force-close edges a last resort


    for i in first_range_tens:
        for j in second_range_tens:
            contrast = apply_brightness_contrast(image.copy(), brightness=i, contrast=j)
            bilat = cv2.bilateralFilter(contrast.copy(),9,75,75) #bilateral filter slower than gaussian but maintains edges better
            edges = layered_edge_detection(bilat.copy())
            sorted_contours = get_contours(edges)

            if len(sorted_contours) > 0:
                max_contour = sorted_contours[0] #largest contour (hopefully the tailgate)

                if cv2.contourArea(max_contour) > int(image_area*.75):
                    full_process.append('contour > 75% of area')

                    hull = cv2.convexHull(max_contour)

                    # drawn_ctrs = cv2.drawContours(contrast.copy(), [hull], -1, (0, 255, 0), 2)

                    masked_image = transparent_tailgate_mask(image, hull)
                    full_process.append('masked')

                    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                    # ax1.imshow(image)
                    # ax1.axis('off')
                    # ax2.imshow(masked_image)
                    # ax2.axis('off')
                    # plt.tight_layout();
                    
                    #print(f'full process [ {(" >>> ").join(full_process)} ]')

                    return masked_image, full_process

                elif cv2.contourArea(max_contour) < int(image_area*.75):
                    if (cv2.contourArea(sorted_contours[0]) + 
                        cv2.contourArea(sorted_contours[1])) > int(image_area*.75):
                        full_process.append('contours added > 75% of area')

                        concat = np.concatenate((sorted_contours[0],sorted_contours[1]), axis=0)

                        hull = cv2.convexHull(concat)

                        # drawn_ctrs = cv2.drawContours(contrast.copy(), [hull], -1, (0, 255, 0), 2)

                        masked_image = transparent_tailgate_mask(image, hull)
                        full_process.append('masked')

                        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                        # ax1.imshow(image)
                        # ax1.axis('off')
                        # ax2.imshow(masked_image)
                        # ax2.axis('off')
                        # plt.tight_layout();

                        # print(f'full process [ {(" >>> ").join(full_process)} ]')

                        return masked_image, full_process

    full_process.append('edge processes tried')

    # Corner Processing         
    for i in first_range_tens:
        for j in second_range_tens:
            contrast = apply_brightness_contrast(image.copy(), brightness=i, contrast=j)
            bilat = cv2.bilateralFilter(contrast.copy(),9,75,75) #bilateral filter slower than gaussian but maintains edges better
            edges = layered_edge_detection(bilat.copy())
            sorted_contours = get_contours(edges)
            corners_array = get_array_of_corners(bilat.copy())

            if len(sorted_contours) > 0 and len(corners_array > 0):
                max_contour = sorted_contours[0] #largest contour (hopefully the tailgate)
   

                max_contour_and_corners = np.concatenate((sorted_contours[0],np.array(corners_array)), axis=0)

                if cv2.contourArea(max_contour_and_corners) > int(image_area*.75):
                    full_process.append('corners + contour > 75% of area')

                    hull = cv2.convexHull(max_contour_and_corners)

                    # drawn_ctrs = cv2.drawContours(contrast.copy(), [hull], -1, (0, 255, 0), 2)

                    masked_image = transparent_tailgate_mask(image, hull)
                    full_process.append('masked')

                    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                    # ax1.imshow(image)
                    # ax1.axis('off')
                    # ax2.imshow(masked_image)
                    # ax2.axis('off')
                    # plt.tight_layout();
                    
                    # print(f'full process [ {(" >>> ").join(full_process)} ]')

                    return masked_image, full_process

    full_process.append('corner processes tried')          

    # If all else fails.... border process
    for i in first_range_tens:
        for j in second_range_tens:
            contrast = apply_brightness_contrast(image.copy(), brightness=i, contrast=j)
            bilat = cv2.bilateralFilter(contrast.copy(),9,75,75)  

            # adding border around image to force-close contours as a last measure
            bordered_edges = border_process(bilat.copy())
            bordered_sorted_contours = get_contours(bordered_edges.copy())
            # drawn_ctrs = cv2.drawContours(bilat.copy(), bordered_sorted_contours, -1, (0, 255, 0), 2)
            # plt.imshow(drawn_ctrs)
               
            if len(bordered_sorted_contours) > 0:
                max_contour = bordered_sorted_contours[0] #largest contour (hopefully the tailgate)

                if cv2.contourArea(max_contour) > int(image_area*.75):
                    full_process.append('bordered >>> contour > 75% of area')

                    hull = cv2.convexHull(max_contour)

                    # drawn_ctrs = cv2.drawContours(contrast.copy(), [hull], -1, (0, 255, 0), 2)

                    masked_image = transparent_tailgate_mask(image, hull)
                    full_process.append('masked')

                    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                    # ax1.imshow(image)
                    # ax1.axis('off')
                    # ax2.imshow(masked_image)
                    # ax2.axis('off')
                    # plt.tight_layout();
                    
                    # print(f'full process [ {(" >>> ").join(full_process)} ]')

                    return masked_image, full_process

                elif cv2.contourArea(max_contour) < int(image_area*.75):
                    if (cv2.contourArea(sorted_contours[0]) + 
                        cv2.contourArea(sorted_contours[1])) > int(image_area*.75):
                        full_process.append('bordered >>> contours added > 75% of area')

                        concat = np.concatenate((sorted_contours[0],sorted_contours[1]), axis=0)

                        hull = cv2.convexHull(concat)

                        # drawn_ctrs = cv2.drawContours(contrast.copy(), [hull], -1, (0, 255, 0), 2)

                        masked_image = transparent_tailgate_mask(image, hull)
                        full_process.append('masked')

                        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                        # ax1.imshow(image)
                        # ax1.axis('off')
                        # ax2.imshow(masked_image)
                        # ax2.axis('off')
                        # plt.tight_layout();

                        # print(f'full process [ {(" >>> ").join(full_process)} ]')

                        return masked_image, full_process

    full_process.append('border processes tried')                

    full_process.append('tailgate not found')
    # print(f'full tailgate process [ {(" >>> ").join(full_process)} ]')
    return False, full_process



def handle_detect_and_mask(image):

    image_area = image.shape[0] * image.shape[1]
    mode_of_image = get_mode(image.copy())

    full_process = [f'mode of image: {mode_of_image}']

    if mode_of_image >= 125:
        first_range_tens = np.arange(0,-111, -10)
        full_process.append('darken process')
    else:
        first_range_tens = np.arange(0,111, 10)
        full_process.append('lighten process')

    second_range_tens = np.arange(0,111, 10)
   

    for i in first_range_tens:
        for j in second_range_tens:
            
            #adapted = adaptive_histogram(image.copy())
            #bilat = cv2.bilateralFilter(image.copy(),9,75,75)
            adjusted = apply_brightness_contrast(image.copy(), brightness=i, contrast=j)
            bilat = cv2.bilateralFilter(adjusted.copy(),11,75,75)

            edges = layered_edge_detection(adjusted.copy())
            sorted_contours = get_contours(edges)

            if len(sorted_contours) > 0:
                max_contour = sorted_contours[0] #largest contour (hopefully the tailgate)

                if cv2.contourArea(max_contour) > int(image_area*.4):
                    # print(f'area of max contour: {cv2.contourArea(max_contour)}')
                    # print(f'.4 * image area: {int(image_area*.4)}')
                    # print(f'image area: {image_area}')
                    full_process.append('contour > 40% of area')

                    hull = cv2.convexHull(max_contour)

                    drawn_ctrs = cv2.drawContours(adjusted.copy(), [hull], -1, (0, 255, 0), 2)

                    masked_image = transparent_handle_mask(image, hull)
                    full_process.append('masked')

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                    ax1.imshow(image)
                    ax1.axis('off')
                    ax2.imshow(masked_image)
                    ax2.axis('off')
                    plt.tight_layout();
                    
                    # print(f'full process [ {(" -> ").join(full_process)} ]')

                    return masked_image, full_process

                elif cv2.contourArea(max_contour) < int(image_area*.40) \
                and len(sorted_contours) > 1:
                    if (cv2.contourArea(sorted_contours[0]) + 
                        cv2.contourArea(sorted_contours[1])) > int(image_area*.4):
                        full_process.append('contours added > 40% of area')

                        concat = np.concatenate((sorted_contours[0],sorted_contours[1]), axis=0)

                        hull = cv2.convexHull(concat)

                        # drawn_ctrs = cv2.drawContours(contrast.copy(), [hull], -1, (0, 255, 0), 2)

                        masked_image = transparent_handle_mask(image, hull)
                        full_process.append('masked')

                        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                        # ax1.imshow(image)
                        # ax1.axis('off')
                        # ax2.imshow(masked_image)
                        # ax2.axis('off')
                        # plt.tight_layout();

                        # print(f'full process [ {(" -> ").join(full_process)} ]')

                        return masked_image, full_process

                elif cv2.contourArea(max_contour) < int(image_area*.4):
                    sorted_contours2 = contours_from_edges_on_contours(image.copy(), sorted_contours)
                    
                    if cv2.contourArea(sorted_contours2[0]) > int(image_area*.4):
                        full_process.append('contour of contour > 40% of area')

                        hull = cv2.convexHull(sorted_contours2[0])

                        drawn_ctrs = cv2.drawContours(adjusted.copy(), [hull], -1, (0, 255, 0), 2)

                        masked_image = transparent_handle_mask(image, hull)
                        full_process.append('masked')

                        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                        # ax1.imshow(image)
                        # ax1.axis('off')
                        # ax2.imshow(masked_image)
                        # ax2.axis('off')
                        # plt.tight_layout();
                        
                        # print(f'full process [ {(" -> ").join(full_process)} ]')

                        return masked_image, full_process
                    
                    elif cv2.contourArea(sorted_contours2[0]) < int(image_area*.4) \
                    and len(sorted_contours2) > 1:
                        if (cv2.contourArea(sorted_contours2[0]) + 
                            cv2.contourArea(sorted_contours2[1])) > int(image_area*.4):
                            full_process.append('contour of contours added together > 40% of area')

                            concat = np.concatenate((sorted_contours2[0],sorted_contours2[1]), axis=0)

                            hull = cv2.convexHull(concat)

                            # drawn_ctrs = cv2.drawContours(contrast.copy(), [hull], -1, (0, 255, 0), 2)

                            masked_image = transparent_handle_mask(image, hull)
                            full_process.append('masked')

                            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                            # ax1.imshow(image)
                            # ax1.axis('off')
                            # ax2.imshow(masked_image)
                            # ax2.axis('off')
                            # plt.tight_layout();

                            # print(f'full process [ {(" -> ").join(full_process)} ]')

                            return masked_image, full_process
                else:
                    pass
            else:
                pass
    # print(f'brightness {i} | contrast {j}')
    full_process.append('handle not found')
    # print(f'full handle process [ {(" -> ").join(full_process)} ]')
    return False, full_process
